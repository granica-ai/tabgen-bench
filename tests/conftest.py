"""A one-table benchmark built from synthetic data, so the chain can be tested
without downloading anything.

``bench`` writes a raw dataset in the bundle's layout (six files in a version
directory), records it the way ``creator_tool.record`` does but without the Hub,
points the package's registry at those records, and stores a TGB split
and its digests.  Every test starts from a benchmark whose chain holds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tabgen_bench import creator_tool, preprocessing, registry

NAME = "toy"
UUID = "00000000-0000-7000-8000-000000000001"
ROWS = 5000


def toy_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    missing = rng.normal(size=ROWS)
    missing[rng.random(ROWS) < 0.1] = np.nan
    color = pd.Series(rng.choice(["red", "green", "blue", "teal"], ROWS)).astype(object)
    color[rng.random(ROWS) < 0.05] = None
    # a timestamp: a date and a time of day to the second, missing now and then
    stamp = (pd.Timestamp("2021-03-01") + pd.to_timedelta(rng.integers(0, 400, ROWS), "D")
             + pd.to_timedelta(rng.integers(0, 86400, ROWS), "s")).to_series(index=range(ROWS))
    stamp[rng.random(ROWS) < 0.05] = pd.NaT
    return pd.DataFrame({
        "num_a": rng.normal(size=ROWS),
        "num_missing": missing,
        "count": rng.integers(0, 50, ROWS),
        "color": color.astype("category"),
        "note": pd.Series([f"note {i % 60}" for i in rng.integers(0, 1000, ROWS)],
                          dtype="string"),
        "when": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, ROWS), "D"),
        "stamp": stamp,
        "flag_const": np.ones(ROWS, dtype="int64"),
        "void": np.full(ROWS, np.nan),            # null in every row
        # 380 labels, about ten rows each: more than the vocabulary limit, with ties
        # all along the cut, yet under a tenth of the training rows
        # limit, with ties everywhere along the cut
        "code": pd.Series([f"c{i}" for i in rng.integers(0, 380, ROWS)]).astype("category"),
        # free text: a different value on nearly every row
        "memo": pd.Series([f"memo {i} about {i % 7}" for i in rng.permutation(ROWS)],
                          dtype="string"),
        # a continuous number the bundle declares a category; the benchmark
        # reads it as a number through a dtype override
        "score": rng.gamma(2.0, 1.5, ROWS),
        "label": pd.Series(rng.choice(["yes", "no"], ROWS)).astype("category"),
    })


DTYPES = {"num_a": "float64", "num_missing": "float64", "count": "int64",
          "color": "category", "note": "string", "when": "datetime64[ns]",
          "stamp": "datetime64[ns]", "flag_const": "int64", "void": "float64", "code": "category", "memo": "string",
          "score": "category", "label": "category"}
OVERRIDES = {NAME: {"score": "float64"}}
# "memo" is one label per row.  What creator_tool.find_dropped_columns says of
# the toy table, hardcoded the way registry.DROPPED_COLUMNS is for the real ones.
DROPPED = {NAME: {"memo": "high cardinality: 4000 labels in 4000 training rows, "
                          "ratio 1.000 above 0.1"}}


def write_raw(raw: Path, frame: pd.DataFrame, dtypes: dict | None = None) -> Path:
    directory = raw / NAME / UUID
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / registry.DOWNLOAD_FILES["table"], index=False)
    payloads = {
        "dtypes": dtypes or DTYPES,
        "dataset": {"unique_name": NAME, "dataset_year": "2024", "domain_str": "testing",
                    "dataset_source": "synthetic", "license": "CC0",
                    "academic_reference_bibtex_key": "toy2024",
                    "original_dataset_source_download_link": "https://example.org/toy"},
        "task": {"target_column_name": "label", "problem_type": "binary_classification",
                 "objective_metric_name": "roc_auc"},
        "splits": {"splits": {"0": {"0": [list(range(3500)), list(range(3500, 4500))]}}},
        "container": {"uuid": UUID, "checksum": "unused", "version_comment": None},
    }
    for key, payload in payloads.items():
        (directory / registry.DOWNLOAD_FILES[key]).write_text(json.dumps(payload))
    return directory


def local_record(raw: Path) -> dict:
    """What ``creator_tool.record`` returns, built from the files alone."""
    directory = raw / NAME / UUID
    files = [{"name": path.name, "path": str(path.relative_to(raw)),
              "size": path.stat().st_size, "sha256": registry.file_sha256(path),
              "git_blob_id": registry.git_blob_id(path),
              "lfs": path.name == registry.DOWNLOAD_FILES["table"],
              "url": f"https://example.org/{path.name}"}
             for path in sorted(directory.iterdir())]
    dtypes = json.loads((directory / registry.DOWNLOAD_FILES["dtypes"]).read_text())
    parquet = pq.ParquetFile(directory / registry.DOWNLOAD_FILES["table"])
    return {
        "table": NAME,
        "origin": {"host": "example.org", "repo_id": "tests/toy", "repo_type": "dataset",
                   "revision": "0" * 40, "revision_date": None, "revision_title": None,
                   "directory": f"{NAME}/{UUID}", "container_uuid": UUID,
                   "version_comment": None, "url": "https://example.org/toy",
                   "repo_license_tag": None},
        "files": files,
        "dataset": {"unique_name": NAME, "task_type": "binary_classification",
                    "target": "label", "rows": parquet.metadata.num_rows,
                    "columns": parquet.metadata.num_columns, "license": "CC0",
                    "original_source": "synthetic"},
        "schema": {"rows": parquet.metadata.num_rows,
                   "columns": [{"name": c, "dtype": dtypes[c],
                                "storage": str(parquet.schema_arrow.field(c).type)}
                               for c in parquet.schema_arrow.names]},
    }


@dataclass
class Bench:
    name: str
    root: Path        # the data root: raw/, prepared/, table_sources/<t>/verified.json
    sources: Path     # the registry the package reads during the test

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def table_file(self) -> Path:
        return self.raw / NAME / UUID / registry.DOWNLOAD_FILES["table"]

    @property
    def record_dir(self) -> Path:
        return self.sources / NAME

    def fresh_root(self, tmp_path: Path, label: str) -> Path:
        """Another data root over the same raw files, for a second build."""
        root = tmp_path / label
        root.mkdir()
        os.symlink(self.raw, root / "raw")
        return root


@pytest.fixture
def bench(tmp_path, monkeypatch) -> Bench:
    root, sources = tmp_path / "data", tmp_path / "table_sources"
    write_raw(root / "raw", toy_frame())
    monkeypatch.setattr(registry, "SOURCES", sources)
    monkeypatch.setattr(registry, "TABLES", (NAME,))
    monkeypatch.setattr(registry, "DTYPE_OVERRIDES", OVERRIDES)
    monkeypatch.setattr(registry, "DROPPED_COLUMNS", DROPPED)
    creator_tool.write_record(sources, local_record(root / "raw"))
    # The creator's two steps: draw and store the split, then fit, store and
    # build the columns (record_columns builds through prepare_table).
    creator_tool.draw_split(NAME, root, val_fraction=0.1, test_fraction=0.1, seed=42)
    creator_tool.record_columns(NAME, root)
    return Bench(NAME, root, sources)
