"""Prepared test-data loading and stable content digests.

Raw-data loading and observation recovery live in ``creator_tool``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _PreparedTaskData:
    """Checked prepared test data; contains no training or raw observations."""

    name: str
    test: pd.DataFrame
    target: str
    feature_groups: tuple[tuple[str, ...], ...]
    schema: dict
    checksums: dict[str, str]
    row_ids_checksum: str

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.test.columns)


def _load_prepared_task_data(name, root):
    """Check only the local test parquet against the shipped benchmark record."""
    from ..datasets import split_file
    from ..registry import benchmark_tables, read_source
    from ..validation import (
        PreparedMismatchError, compute_dataframe_digest, load_column_spec, load_split_indices,
    )

    if name not in benchmark_tables():
        raise KeyError(f"Unknown TabGenBench table {name!r}")
    # Use shipped records as the reference for the local prepared test file.
    schema = load_column_spec(name)
    source = read_source(name)
    path = split_file(root, name, "test")
    if not path.is_file():
        raise FileNotFoundError(f"Prepared test data is required: {path}")
    test = pd.read_parquet(path)
    # Stored mask coordinates depend on this exact column order and row count.
    if (tuple(test.columns) != tuple(schema["column_order"])
            or len(test) != source["prepared"]["rows"]["test"]):
        raise PreparedMismatchError(f"{name}: prepared test shape/schema differs from the benchmark")
    # Matching shape alone cannot detect changed values or reordered test rows.
    digest = compute_dataframe_digest(test)
    if digest != source["prepared"]["digests"]["test"]:
        raise PreparedMismatchError(f"{name}: prepared test content checksum differs from the benchmark")
    # This reads the split metadata shipped with the package, not local raw data.
    _, positions = load_split_indices(name)
    # Group columns and missing indicators by their original source column
    # so task construction can move each complete group into inputs or targets.
    groups = {}
    for column in schema["columns"]:
        raw = column.get("indicator_of") or column.get("part_of") or column["name"]
        groups.setdefault(raw, []).append(column["name"])
    # The mask reader checks content, schema and original test-row identity
    # against its manifest before applying any stored coordinates.
    return _PreparedTaskData(
        name, test.reset_index(drop=True), source["dataset"]["target"],
        tuple(tuple(group) for group in groups.values()), schema,
        {"test": digest, "schema": _json_digest(schema)},
        _array_digest(np.asarray(positions["test"], dtype=np.int64)),
    )
