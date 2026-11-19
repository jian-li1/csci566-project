"""Small, durable JSON files and safe interruption recovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path):
    """Repair only an incomplete final write; reject any other corrupt record."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    records, offset = [], 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"Corrupt results journal: {path}, line {index + 1}")
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}")
        records.append(value)
        offset += len(line)
    if records and raw and not raw.endswith(b"\n") and offset == len(raw):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return records
