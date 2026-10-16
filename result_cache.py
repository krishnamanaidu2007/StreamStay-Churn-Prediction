"""Small process-local helpers for deterministic StreamStay result caches."""

import hashlib
import os
import uuid
from pathlib import Path
from typing import Callable


def source_signature(*paths: str) -> str:
    """Hash the exact source files that determine one computation's output."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def result_version(data_fingerprint: str, computation_signature: str) -> str:
    """Short, deterministic token safe for a static chart filename."""
    return hashlib.sha256(
        f"{data_fingerprint}:{computation_signature}".encode("utf-8")
    ).hexdigest()[:16]


def versioned_chart_filename(filename: str, version: str) -> str:
    """Keep the chart's readable name while binding it to one result version."""
    stem, extension = os.path.splitext(filename)
    return f"{stem}-{version}{extension}"


def chart_files_exist(charts_dir: str, filenames: list[str]) -> bool:
    return all(os.path.isfile(os.path.join(charts_dir, filename)) for filename in filenames)


def save_chart_atomically(destination: str, save: Callable[[str], None]) -> None:
    """Publish a finished PNG atomically; readers never receive a partial file."""
    temporary = f"{destination}.{uuid.uuid4().hex}.tmp.png"
    try:
        save(temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
