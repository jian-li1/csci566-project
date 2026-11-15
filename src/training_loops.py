import math
import os
import time

import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup

from run_summary import RunSummary, format_duration

import sys

# Shared compute budget (compute_budget/ at the repo root), same module GRPO uses.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "compute_budget"))
from compute_budget import ComputeBudget


class TrainingLoops:
    def __init__(
        self,
        model,
        dataset=None,
        training_portion=0.8,
        validation_portion=0.2,
        batch_size=1,
        epochs=5,
        lr=1e-4,
        gradient_accumulation_steps=16,
        output_dir="./checkpoints",
        trainset=None,
        valset=None,
        summary=None,
        log_every=None,
        checkpoint_every=50,
        start_epoch=0,
        start_step=0,
        budget=None,
        budget_stage=None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        self.checkpoint_every = checkpoint_every
        self.start_epoch = start_epoch
        os.makedirs(self.output_dir, exist_ok=True)

        # Exactly one source: either a whole dataset to split, or a train/val pair.
        if dataset is not None and (trainset is not None or valset is not None):
            raise ValueError("Provide a whole dataset OR a trainset/valset pair, not both")
        if trainset is not None or valset is not None:
            if trainset is None or valset is None:
                raise ValueError("Both trainset and valset are required")
            if len(trainset) == 0 or len(valset) == 0:
                raise ValueError("Train and validation datasets must both be nonempty")
            self.train_set = trainset
            self.validation_set = valset
            self.total_dataset_size = len(trainset) + len(valset)
        elif dataset is not None:
            # One dataset, with a deterministic train/validation split.
            if len(dataset) < 2:
                raise ValueError("Need at least 2 examples to create train and validation sets")
            if not (0 < training_portion < 1 and 0 < validation_portion < 1):
                raise ValueError("Train/validation fractions must each be between 0 and 1")
            if not math.isclose(training_portion + validation_portion, 1.0, abs_tol=1e-6):
                raise ValueError("Train and validation fractions must sum to 1")
            # Integer split ensures both splits are nonempty, even for small datasets.
            train_count = max(1, min(len(dataset) - 1, int(len(dataset) * training_portion)))
            val_count = len(dataset) - train_count
            self.train_set, self.validation_set = random_split(
                dataset, [train_count, val_count], generator=torch.Generator().manual_seed(42)
            )
            self.total_dataset_size = len(dataset)
        else:
            raise ValueError("Provide either dataset or trainset")

        if len(self.train_set) == 0:
            raise ValueError("Training dataset is empty")
        self.dataset = self.train_set  # For sample_train() compatibility.
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        # Validation is mandatory, so always create its DataLoader.
        self.validation_loader = DataLoader(
            self.validation_set, batch_size=self.batch_size, shuffle=False
        )

        trainable_parameters = [p for p in self.model.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters; check the LoRA/model configuration")
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr)

        updates_per_epoch = math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        total_updates = updates_per_epoch * self.epochs
        warmup_steps = int(0.03 * total_updates)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_updates,
        )
        if start_epoch and start_step != start_epoch * updates_per_epoch:
            raise ValueError("Resume checkpoint step does not match the current dataset and accumulation settings")
        self.global_step = start_step
        if start_step:
            # Keep the original three-epoch LR schedule; optimizer moments are unavailable.
            resumed_lrs = [base * schedule(start_step) for base, schedule in
                           zip(self.scheduler.base_lrs, self.scheduler.lr_lambdas)]
            for group, resumed_lr in zip(self.optimizer.param_groups, resumed_lrs):
                group["lr"] = resumed_lr
            self.scheduler.last_epoch = start_step
            self.scheduler._step_count = start_step + 1
            self.scheduler._last_lr = resumed_lrs
        self.total_updates = total_updates
        self._last_saved_step = -1
        self._last_checkpoint_dir = None

        # Default: about 20 progress lines per epoch, whatever the dataset size
        if log_every is not None and log_every <= 0:
            raise ValueError("log_every must be positive")
        self.log_every = log_every or max(1, updates_per_epoch // 20)

        config = {
            "Model": self.model.model_name,
            "Total dataset size": self.total_dataset_size,
            "Training samples": len(self.train_set),
            "Validation samples": len(self.validation_set),
            "Batch size": self.batch_size,
            "Gradient accumulation": self.gradient_accumulation_steps,
            "Effective batch size": self.batch_size * self.gradient_accumulation_steps,
            "Epochs": self.epochs,
            "Learning rate": self.lr,
            "Train batches per epoch": len(self.train_loader),
            "Optimizer updates/epoch": updates_per_epoch,
            "Total optimizer updates": total_updates,
            "Warmup steps": warmup_steps,
            "Log every": f"{self.log_every} optimizer steps",
            "Checkpoint every": f"{self.checkpoint_every} steps" if self.checkpoint_every else "epoch end",
            "Starting epoch": self.start_epoch + 1,
            "Starting optimizer step": self.global_step,
        }
        print("\n===== Training Configuration =====")
        for label, value in config.items():
            print(f"{label + ':':<26}{value}")
        print("==================================\n")

        self.summary = summary or RunSummary(self.output_dir)
        self.summary.total_epochs = self.epochs
        self.summary.set_section("Training config", config)
        self.summary.write()

        # Compute tracking -> <output_dir>/compute.json (tokens, FLOPs, GPU-hours).
        # budget=None tracks without stopping; budget_stage="sft_then_grpo:sft" uses 25% of B.
        self.budget_tracker = ComputeBudget(
            self.output_dir, mode="sft", budget=budget, lora=True, stage=budget_stage
        )
        self._window_tokens = 0
        self._budget_reached = False

    def sample_train(self):
        sample = self.train_set[0]
        for epoch in range(2):
            total_loss = 0.0
            for i in range(20):
                loss = self.model.train_step(sample["question"], sample["answer"])
                total_loss += float(loss)
                print(f"Sample: Step {i + 1}: {float(loss):.4f}")
            print(f"Sample: Average loss: {total_loss / 20:.4f}")

    def validate(self):
        print("\nRunning validation...")
        self.model.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in self.validation_loader:
                questions = batch["question"]
                answers = batch["answer"]
                systems = batch.get("system", [""] * len(questions))
                loss = self.model.calculate_batch_loss(questions=questions,answers=answers,systems=systems)
                count = len(questions)
                total_loss += loss.item() * count
                total_samples += count
        return total_loss / total_samples

    def train(self):
        try:
            self._train()
        except BaseException as e:
            # Record the failure (including Ctrl+C / Slurm cancel) before re-raising
            self.summary.finish(f"failed: {type(e).__name__}: {e}")
            raise
        self.summary.finish("completed")

    def _train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\n{'=' * 50}\nStarting Epoch {epoch + 1}/{self.epochs}\n{'=' * 50}")
            epoch_start = time.time()
            self.model.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_examples = 0

            # Loss of the current accumulation window (one optimizer update)
            window_loss = 0.0
            window_count = 0
            # Update losses since the last progress line
            log_loss = 0.0
            log_updates = 0

            for batch_idx, batch in enumerate(self.train_loader):
                questions = batch["question"]
                answers = batch["answer"]
                systems = batch.get("system", [""] * len(questions))

                loss = self.model.calculate_batch_loss(questions= questions, answers= answers, systems=systems)
                current_batch_size = len(questions)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size
                window_loss += loss.item() * current_batch_size
                window_count += current_batch_size
                self._window_tokens += getattr(self.model, "last_num_tokens", 0)

                # Scale by actual example count in this accumulation window;
                # this also handles the partial window at the end of an epoch.
                group_start_batch = (batch_idx // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
                remaining_examples = len(self.train_set) - group_start_batch * self.batch_size
                window_examples = min(self.gradient_accumulation_steps * self.batch_size, remaining_examples)
                scaled_loss = loss * (current_batch_size / window_examples)
                scaled_loss.backward()

                should_update = (batch_idx + 1) % self.gradient_accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == len(self.train_loader)
                if should_update or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    if self.budget_tracker.step(tokens=self._window_tokens):
                        self._budget_reached = True
                    self._window_tokens = 0

                    current_lr = self.scheduler.get_last_lr()[0]
                    update_loss = window_loss / window_count
                    self.summary.log_metric(epoch + 1, self.global_step, "train", update_loss, current_lr)
                    window_loss = 0.0
                    window_count = 0

                    log_loss += update_loss
                    log_updates += 1
                    if self.global_step % self.log_every == 0 or is_last_batch:
                        print(
                            f"Epoch {epoch + 1}/{self.epochs} | "
                            f"Step {self.global_step}/{self.total_updates} | "
                            f"Loss {log_loss / log_updates:.4f} (avg of {log_updates} steps) | "
                            f"LR {current_lr:.2e} | "
                            f"{format_duration(time.time() - epoch_start)} elapsed"
                        )
                        log_loss = 0.0
                        log_updates = 0

                    if self.checkpoint_every and self.global_step % self.checkpoint_every == 0:
                        self.save_checkpoint(epoch)

                    if self._budget_reached:
                        print("Compute budget reached; finishing this epoch's validation and stopping.")
                        break

            average_train_loss = total_loss / total_examples
            validation_loss = self.validate()
            epoch_seconds = time.time() - epoch_start
            print("\n" + "=" * 50)
            print(f"Epoch {epoch + 1}/{self.epochs} Complete")
            print(f"Average training loss:   {average_train_loss:.4f}")
            print(f"Average validation loss: {validation_loss:.4f}")
            print(f"Global optimizer steps:  {self.global_step}")
            print(f"Epoch time:              {format_duration(epoch_seconds)}")
            print("=" * 50)
            self.save_checkpoint(epoch)

            self.summary.log_metric(epoch + 1, self.global_step, "train_epoch", average_train_loss)
            self.summary.log_metric(epoch + 1, self.global_step, "val", validation_loss)
            self.summary.add_epoch(
                epoch=epoch + 1,
                train_loss=average_train_loss,
                val_loss=validation_loss,
                steps=self.global_step,
                seconds=epoch_seconds,
                checkpoint=self._last_checkpoint_dir,
            )
            if self._budget_reached:
                break

    def save_checkpoint(self, epoch):
        if self._last_saved_step == self.global_step:
            return self._last_checkpoint_dir
        # The run folder already names the model and dataset
        checkpoint_dir = os.path.join(
            self.output_dir, "checkpoints", f"epoch{epoch + 1}_step{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.model.save_pretrained(checkpoint_dir)
        self.model.tokenizer.save_pretrained(checkpoint_dir)
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        self._last_saved_step = self.global_step
        self._last_checkpoint_dir = checkpoint_dir
        print(f"Checkpoint saved at: {checkpoint_dir}")
        return checkpoint_dir
