"""The datasets on disk: reading a raw table, and reading a prepared one.

**Raw.**  A downloaded dataset is a directory of six files
(:data:`tabgen_bench.registry.DOWNLOAD_FILES`): the table, its declared dtypes, the
dataset and task metadata, the benchmark's folds and a container record.
:func:`read_raw` reads them, and reads the table *as declared*: column kinds
come from ``dtypes.json``, not from what the parquet file happens to store.
The two differ on 30-odd BeyondArena tables (a column stored as float64 and
declared ``category``, yes/no columns stored as int64), and the declared
dtype is the one every published number on these tasks was computed with, so
:func:`apply_dtypes` casts to it before anything reads the frame.
``string``, the bundle's free-text dtype, is a categorical here: there is no
text channel to send it down.

**Loading.**  :func:`load_table` is what a user of the benchmark calls: the
prepared train / validation / test splits of a table as DataFrames.  It
needs nothing done first: a table that is not under the data root is
downloaded and built, and one that is there is checked against the
benchmark's content digests before it is returned (a check that is reused
while the files are untouched).  :func:`load_spec` gives the table's
columns, kinds, vocabularies, target and task the same way; :func:`load_raw`
the raw table, typed as declared.

**Prepared.**  :mod:`tabgen_bench.preprocessing` writes
``<data root>/prepared/<table>/`` -- ``train`` / ``validation`` / ``test``
parquet files and ``spec.json`` (columns, kinds, vocabularies, task, target,
and how the split was made) -- and ``prepared/manifest.json``, one summary
entry per table.  The split itself is the package's
``tgb_split_indices.parquet``; nothing of it is written here.  This module
reads them back (:func:`read_manifest`, :func:`read_spec`) and reports on a
prepared corpus:

    python -m tabgen_bench.datasets
    python -m tabgen_bench.datasets --datasets sdss_17 kick
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .registry import (
    DATA_ROOT,
    DOWNLOAD_FILES,
    SPLITS,
    benchmark_tables,
    default_data_root,
)
from .validation import verify_prepared_dataframes


# --------------------------------------------------------------------------
# Raw tables
# --------------------------------------------------------------------------

def dataset_dir(name: str, raw: Path) -> Path:
    """The directory holding ``name``'s six files, whichever layout it uses."""
    matches = sorted(p.parent for p in (Path(raw) / name).rglob(DOWNLOAD_FILES["table"]))
    if not matches:
        raise FileNotFoundError(f"no {DOWNLOAD_FILES['table']} under {Path(raw) / name}")
    if len(matches) > 1:
        raise ValueError(f"{name} has {len(matches)} versions under {raw}; "
                         f"expected one")
    return matches[0]


def read_raw(name: str, raw: Path):
    """``(frame, dtypes, task, dataset_metadata)`` for one dataset.

    The frame is cast to the dtypes the bundle declares before anything reads
    it; see :func:`apply_dtypes`.  This is the table as BeyondArena defines
    it.  The few columns the benchmark prepares under another dtype
    (:data:`tabgen_bench.registry.DTYPE_OVERRIDES`) are recast by the
    preparation, not here.
    """
    directory = dataset_dir(name, raw)
    dtypes = json.loads((directory / DOWNLOAD_FILES["dtypes"]).read_text())
    task = json.loads((directory / DOWNLOAD_FILES["task"]).read_text())
    metadata = json.loads((directory / DOWNLOAD_FILES["dataset"]).read_text())
    frame = apply_dtypes(pd.read_parquet(directory / DOWNLOAD_FILES["table"]), dtypes)
    return frame, dtypes, task, metadata


def kind_of(dtype: str) -> str:
    """Classify a declared pandas dtype.

    ``numerical`` and ``categorical`` are the two kinds the schema knows;
    ``datetime`` marks a numerical column ``prepare`` has to convert to days
    first.  ``string``, the benchmark's text dtype, is categorical: there is
    no text channel to send it down.  An unrecognised dtype is an error rather
    than a guess: the bundle declares fifteen distinct dtypes and a new one
    should be looked at.
    """
    if dtype.startswith(("datetime", "period")):
        return "datetime"
    if dtype in ("category", "string", "bool", "boolean", "object"):
        return "categorical"
    if dtype.startswith(("int", "uint", "float", "Int", "UInt", "Float")):
        return "numerical"
    raise ValueError(f"unrecognised declared dtype {dtype!r}")


def apply_dtypes(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    """Cast the frame to its declared dtypes, the way Data Foundry loads it.

    Only the columns whose stored dtype differs are touched.  The column sets
    have to agree exactly: a declaration the table does not carry, or a
    column it does not declare, is a corrupt download.
    """
    declared, stored = set(dtypes), set(frame.columns)
    if declared != stored:
        raise ValueError(
            f"dtypes.json and the table disagree on columns: "
            f"only declared {sorted(declared - stored)}, "
            f"only stored {sorted(stored - declared)}")
    frame = frame[list(dtypes)]
    for column, dtype in dtypes.items():
        if str(frame[column].dtype) == dtype:
            continue
        if dtype == "category":
            frame[column] = frame[column].astype("category")
        elif dtype.startswith("datetime"):
            frame[column] = pd.to_datetime(frame[column])
        elif dtype.startswith("period"):
            frame[column] = frame[column].astype(dtype)
        else:
            frame[column] = frame[column].astype(dtype)
    return frame


# --------------------------------------------------------------------------
# Prepared tables
# --------------------------------------------------------------------------

def table_dir(root: Path, table: str) -> Path:
    return Path(root) / "prepared" / table


def split_file(root: Path, table: str, split: str) -> Path:
    return table_dir(root, table) / f"{split}.parquet"


def spec_file(root: Path, table: str) -> Path:
    return table_dir(root, table) / "spec.json"


def read_manifest(data_root: Path = DATA_ROOT) -> dict:
    path = Path(data_root) / "prepared" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run python -m tabgen_bench.preprocessing first")
    return json.loads(path.read_text())


def read_spec(name: str, data_root: Path = DATA_ROOT) -> dict:
    path = spec_file(data_root, name)
    if not path.exists():
        raise FileNotFoundError(f"{name} is not in the prepared corpus ({path})")
    return json.loads(path.read_text())


def ensure_prepared(name: str, data_root: Path) -> dict:
    """Have ``name`` prepared and checked under ``data_root``; return its spec.

    A table with no complete prepared folder (the spec and the three
    parquet files) is downloaded if its raw files are absent and built
    through the whole reproducibility chain
    (:func:`tabgen_bench.preprocessing.prepare`), which checks its digests
    as it finishes.  A table that is there is held to the benchmark's content
    digests (:func:`tabgen_bench.validation.verify_prepared_dataframes`; a
    check that is reused while the files are untouched), and a
    :class:`~tabgen_bench.validation.PreparedMismatchError` is raised if
    they are not the benchmark's data.
    """
    if name not in benchmark_tables():
        raise KeyError(f"{name!r} is not a benchmark table; see tabgen_bench.list_tables()")
    files = [spec_file(data_root, name)] + [split_file(data_root, name, s) for s in SPLITS]
    if not all(path.exists() for path in files):
        from .preprocessing import prepare

        prepare(data_root, [name], force=True)
    else:
        verify_prepared_dataframes(name, data_root)
    return read_spec(name, data_root)


def load_spec(name: str, *, data_root: Path | None = None) -> dict:
    """The spec of a table, downloading and building it first if need be.

    ``columns`` lists every prepared column with its ``kind`` (``numerical``
    or ``categorical``), its categories or fill, and a description;
    ``column_order`` is the order of the prepared columns; ``target`` and
    ``task_type`` name the prediction task; ``dropped`` the raw columns left
    out and why; and ``digests`` holds the content digest of each split.
    The table is prepared and checked as :func:`load_table` prepares and
    checks it.
    """
    return ensure_prepared(name, Path(data_root or default_data_root()))


def load_table(name: str, split: str | None = None, *, data_root: Path | None = None):
    """The prepared splits of ``name`` as DataFrames.

    Returns ``{"train": ..., "validation": ..., "test": ...}``, or the one
    DataFrame when ``split`` names a split.  Numerical columns are float64 and
    categorical columns hold their labels as strings; there are no nulls.

    Nothing has to be run first.  A table that is not under the data root is
    downloaded and built; one that is there is held to the benchmark's
    content digests before anything is returned, whichever split is asked
    for, and a :class:`~tabgen_bench.validation.PreparedMismatchError` is
    raised if its files are not the benchmark's data.  The check reads
    every value once and is then reused while the files are untouched
    (:func:`ensure_prepared`).
    """
    root = Path(data_root or default_data_root())
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    ensure_prepared(name, root)
    wanted = SPLITS if split is None else (split,)
    frames = {part: pd.read_parquet(split_file(root, name, part)) for part in wanted}
    return frames if split is None else frames[split]


def load_raw(name: str, *, data_root: Path | None = None) -> pd.DataFrame:
    """The raw table of ``name``, cast to the dtypes BeyondArena declares.

    This is the table as downloaded, before any split or cleaning: nulls,
    dates and free text are still in it.  It is not checked against the
    benchmark's records; ``tabgen_bench.verify_tables`` does that.
    """
    return read_raw(name, Path(data_root or default_data_root()) / "raw")[0]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="tables to report on (default: every prepared table)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    manifest = read_manifest(args.data_root)
    names = args.datasets or list(manifest["tables"])
    entries = {name: manifest["tables"][name] for name in names}
    widths = [entry["columns"] for entry in entries.values()]
    print(json.dumps({
        "prepared": len(entries),
        "columns": {"min": min(widths), "max": max(widths),
                    "total": sum(widths)},
        "rows": {s: sum(e["rows"][s] for e in entries.values()) for s in SPLITS},
        "max_cardinality": max(e["max_cardinality"] for e in entries.values()),
        "max_categories": sorted({e["max_categories"] for e in entries.values()}),
        "tables": entries,
    }, indent=1))


if __name__ == "__main__":
    main()
