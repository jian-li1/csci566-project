"""Human-readable summary.txt and machine-readable metrics.csv for one run.

summary.txt is rewritten after every epoch, so a run that crashes or hits
the Slurm time limit still leaves an up-to-date summary behind.
"""

import csv
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

import torch


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


def git_info():
    """Short commit hash plus whether tracked files have uncommitted edits."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=src_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=src_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit} (dirty: {'yes' if dirty else 'no'})"


INDEX_COLUMNS = [
    "finished", "dataset", "model", "run", "epochs_done", "lr", "samples",
    "best_epoch", "best_val_loss", "final_val_loss", "status", "path",
]


class RunSummary:
    def __init__(self, output_dir, index_path=None, index_info=None):
        """index_path: shared CSV that gets one row per finished run.
        index_info: fixed columns for that row (dataset, model, lr, samples).
        """
        self.output_dir = output_dir
        self.index_path = index_path
        self.index_info = index_info or {}
        os.makedirs(output_dir, exist_ok=True)
        self.summary_path = os.path.join(output_dir, "summary.txt")
        self.metrics_path = os.path.join(output_dir, "metrics.csv")
        self.start_time = time.time()

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            gpu = "Apple MPS"
        else:
            gpu = "none (CPU)"

        # Sections print in insertion order; Results is always last
        self.sections = {
            "Run": {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Slurm job": os.environ.get("SLURM_JOB_ID", "none (local run)"),
                "Git commit": git_info(),
                "GPU": gpu,
                "Command": shlex.join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
            }
        }
        self.total_epochs = None
        self.epochs = []
        self.best = None
        self.status = "starting"

        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "split", "loss", "lr", "elapsed_s"])

    def set_section(self, title, rows):
        """Add or replace a section of label -> value rows."""
        self.sections[title] = dict(rows)

    def log_metric(self, epoch, step, split, loss, lr=None):
        """Append one row to metrics.csv.

        split is "train" (one optimizer update), "train_epoch" (epoch
        average) or "val" (epoch validation loss).
        """
        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                step,
                split,
                f"{loss:.6f}",
                "" if lr is None else f"{lr:.8g}",
                f"{time.time() - self.start_time:.1f}",
            ])

    def add_epoch(self, epoch, train_loss, val_loss, steps, seconds, checkpoint):
        self.epochs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "steps": steps,
            "seconds": seconds,
            "checkpoint": checkpoint,
        })
        if self.best is None or val_loss < self.best["val_loss"]:
            self.best = self.epochs[-1]
            self._link_best(checkpoint)
        self.status = f"running (epoch {epoch}/{self.total_epochs})"
        self.write()

    def finish(self, status):
        self.status = status
        self.write()
        if self.index_path:
            self._append_index()

    def _link_best(self, checkpoint):
        """Point <run>/best at the lowest-validation-loss checkpoint."""
        if not checkpoint:
            return
        link = os.path.join(self.output_dir, "best")
        tmp_link = link + ".tmp"
        if os.path.lexists(tmp_link):
            os.remove(tmp_link)
        # Relative target, so the run folder can be moved or copied
        os.symlink(os.path.relpath(checkpoint, self.output_dir), tmp_link)
        os.replace(tmp_link, link)

    def _append_index(self):
        last = self.epochs[-1] if self.epochs else None
        row = {
            "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run": os.path.basename(os.path.normpath(self.output_dir)),
            "epochs_done": f"{len(self.epochs)}/{self.total_epochs}",
            "best_epoch": self.best["epoch"] if self.best else "",
            "best_val_loss": f"{self.best['val_loss']:.4f}" if self.best else "",
            "final_val_loss": f"{last['val_loss']:.4f}" if last else "",
            "status": self.status,
            "path": os.path.relpath(self.output_dir, os.path.dirname(self.index_path)),
            **self.index_info,
        }
        new_file = not os.path.exists(self.index_path)
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _relative(self, path):
        return os.path.relpath(path, self.output_dir) if path else "none"

    def _results_lines(self):
        lines = ["Epoch | Train loss | Val loss |  Val ppl | Steps |     Time"]
        for e in self.epochs:
            # Clamp before exp so a diverged run doesn't overflow
            ppl = math.exp(min(e["val_loss"], 50))
            lines.append(
                f"{e['epoch']:>5} | {e['train_loss']:>10.4f} | {e['val_loss']:>8.4f} | "
                f"{ppl:>8.3f} | {e['steps']:>5} | {format_duration(e['seconds']):>8}"
            )
        if not self.epochs:
            lines.append("(no epoch finished yet)")
        lines.append("")

        if torch.cuda.is_available():
            peak = f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
        else:
            peak = "n/a (no CUDA)"

        rows = {}
        if self.best is not None:
            rows["Best epoch"] = (
                f"{self.best['epoch']} (val loss {self.best['val_loss']:.4f}) -> "
                f"{self._relative(self.best['checkpoint'])}"
            )
            rows["Final ckpt"] = self._relative(self.epochs[-1]["checkpoint"])
        rows["Total time"] = format_duration(time.time() - self.start_time)
        rows["Peak GPU mem"] = peak
        rows["Status"] = self.status
        lines += self._rows_lines(rows)
        return lines

    @staticmethod
    def _rows_lines(rows):
        width = max(len(label) for label in rows) + 2
        return [f"{label + ':':<{width}}{value}" for label, value in rows.items()]

    def write(self):
        lines = []
        for title, rows in self.sections.items():
            lines.append(f"===== {title} =====")
            lines += self._rows_lines(rows) if rows else []
            lines.append("")
        lines.append("===== Results =====")
        lines += self._results_lines()

        # Write to a temp file first so a crash mid-write never leaves it half-written
        tmp_path = self.summary_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp_path, self.summary_path)
