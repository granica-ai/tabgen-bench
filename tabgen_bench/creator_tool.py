"""Maintainer's tool: write the origin records the benchmark ships.

For every table, ``tabgen_bench/table_sources/<table>/`` pins which bytes the
benchmark is built from (:mod:`tabgen_bench.registry` describes the folder).
This module writes the four origin sections of that folder -- ``origin.json``,
``files.json``, ``dataset.json`` and ``schema.json`` -- from the Hugging Face
Hub's file listing at one commit and the files on disk.  A record is
cross-checked, not copied: every local file's size, git blob id (plain
files) or sha256 (LFS files) must equal what the Hub lists, or :func:`record`
stops.  The check guards the record and leaves no file of its own;
``verified.json``, which describes one machine's download, is written by
:mod:`tabgen_bench.validation` under the user's data root.

The TGB sections of a folder are written here too: ``tgb_split.json`` with
``tgb_split_indices.parquet`` by :func:`draw_split`, and ``tgb_columns.json``
with ``tgb_prepared.json`` by :func:`record_columns`, which fits the column
specification (:func:`fit_column_spec`, the benchmark's fit mode: a build
only ever applies what it stored).

It also holds the rule behind :data:`tabgen_bench.registry.DROPPED_COLUMNS`:
``--dropped-columns`` reads every table's raw rows and its TGB training split
and prints the columns the rule drops, as the literal the registry stores.

Benchmark users never run this.  The raw files of a table must be on disk
under ``--raw`` first; :func:`tabgen_bench.download.download` fetches only
recorded tables, so a new table's files are placed by other means (the
``huggingface_hub`` client, say).

    python -m tabgen_bench.creator_tool                       # refresh the record of every table in TABLES
    python -m tabgen_bench.creator_tool --datasets <name>     # record a new table (then list it in TABLES)
    python -m tabgen_bench.creator_tool --raw <data root>/raw
    python -m tabgen_bench.creator_tool --revision <commit> --datasets sdss_17
    python -m tabgen_bench.creator_tool --dropped-columns --data-root <data root>
    python -m tabgen_bench.creator_tool --draw-split --datasets <table> --data-root <data root>
    python -m tabgen_bench.creator_tool --record-columns --datasets <table> --data-root <data root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib

import numpy as np
import pandas as pd
from pathlib import Path

from .datasets import dataset_dir, kind_of
from .preprocessing import (
    EPOCH,
    MAX_CATEGORIES,
    MISSING,
    MISSING_INDICATOR_MIN_FRAC,
    OTHER,
    TIMESTAMP_PARTS,
    _convert_to_string_labels,
    _days,
    _hours,
    _numbers,
    _text_order,
    apply_dtype_overrides,
    encode_splits,
    load_raw_splits,
    prepare_table,
)
from .registry import (
    DATA_ROOT,
    DOWNLOAD_FILES,
    DROPPED_COLUMNS,
    dropped_columns,
    HF_HOST,
    HF_REPO,
    SOURCE_SECTIONS,
    SOURCES,
    SPLITS,
    TABLES,
    TGB_SECTIONS,
    TGB_SPLIT_INDICES,
    __version__,
    file_sha256,
    git_blob_id,
    read_source,
    section_text,
    source_dir,
    split_indices_path,
)
from .validation import DIGEST_LABEL, check_partition, compute_dataframe_digest


# --------------------------------------------------------------------------
# What is already recorded
# --------------------------------------------------------------------------

def recorded_tables(out: Path = SOURCES) -> list[str]:
    """The tables that already have a record under ``out``."""
    return sorted(path.parent.name for path in Path(out).glob("*/origin.json"))


def read_record(out: Path, name: str) -> dict | None:
    """The record of ``name`` reassembled from its folder, or None without one."""
    directory = out / name
    if not (directory / "origin.json").exists():
        return None
    entry = {section: json.loads((directory / file).read_text())
             for section, file in SOURCE_SECTIONS.items()}
    origin = dict(entry["origin"])
    entry["table"] = origin.pop("table")
    entry["origin"] = origin
    return entry


def cached_revision(name: str, raw: Path) -> str | None:
    """The commit the download cache recorded for ``name``, if it is unanimous."""
    cache = raw / ".cache" / "huggingface" / "download" / name
    commits = {meta.read_text().splitlines()[0].strip()
               for meta in cache.rglob("*.metadata") if meta.read_text().strip()}
    return commits.pop() if len(commits) == 1 else None


# --------------------------------------------------------------------------
# The columns the benchmark leaves out
# --------------------------------------------------------------------------

# A categorical column with more distinct labels than this share of the TGB
# training split's rows is dropped.  The training split is what a generator
# sees, so it is where a column is one label per row or is not; judged there,
# an identifier whose rows are spread over the splits (a customer with a
# dozen rows, say) reads as the identifier it is, where on the whole table it
# would hide just under the ratio.
MAX_CARDINALITY_RATIO = 0.1


def high_cardinality_columns(train, dtypes: dict, target: str,
                             max_ratio: float = MAX_CARDINALITY_RATIO) -> dict:
    """Categorical columns that are too close to one label per row, judged on ``train``.

    Returns ``{column: reason}`` for every categorical column, the target
    excepted, whose distinct non-null labels in ``train`` exceed ``max_ratio``
    of its rows.  Rows with a missing label count in the denominator.
    ``max_ratio`` of 0 drops nothing.
    """

    high = {}
    if not max_ratio or not len(train):
        return high
    for column, dtype in dtypes.items():
        if kind_of(dtype) != "categorical" or column == target:
            continue
        distinct = int(_convert_to_string_labels(train[column]).nunique(dropna=True))
        if distinct > max_ratio * len(train):
            high[column] = (f"high cardinality: {distinct} labels in {len(train)} training "
                            f"rows, ratio {distinct / len(train):.3f} above {max_ratio:g}")
    return high


def find_dropped_columns(names=None, data_root: Path = DATA_ROOT,
                         max_ratio: float = MAX_CARDINALITY_RATIO) -> dict:
    """What :data:`registry.DROPPED_COLUMNS` should hold, computed from the data.

    For each table the raw rows are read from ``<data_root>/raw`` and the
    TGB training split taken from the stored split indices; the rule is
    :func:`high_cardinality_columns`.  Returns ``{table: {column: reason}}``
    for the tables with a column to drop.
    """
    from .datasets import read_raw
    from .validation import load_split_indices

    found = {}
    for name in names or TABLES:
        frame, dtypes, task, *_ = read_raw(name, Path(data_root) / "raw")
        frame, dtypes = apply_dtype_overrides(name, frame, dtypes)   # as the build reads it
        _, positions = load_split_indices(name)
        train = frame.iloc[positions["train"]]
        high = high_cardinality_columns(train, dtypes, task["target_column_name"], max_ratio)
        if high:
            found[name] = high
    return found


def dropped_columns_literal(found: dict) -> str:
    """``found`` as the Python literal ``registry.DROPPED_COLUMNS`` is written as."""
    text = "DROPPED_COLUMNS = {\n"
    for table, columns in found.items():
        text += f'    "{table}": {{\n'
        for column, why in columns.items():
            text += f'        "{column}":\n            "{why}",\n'
        text += "    },\n"
    return text + "}\n"


# --------------------------------------------------------------------------
# Writing a record
# --------------------------------------------------------------------------

def record(name: str, raw: Path, api, revision: str, commit: dict, repo_license) -> dict:
    listing = [entry for entry in api.list_repo_tree(
        HF_REPO, path_in_repo=name, repo_type="dataset", revision=revision, recursive=True)
        if type(entry).__name__ == "RepoFile" and "text_cache" not in entry.path]
    if not listing:
        raise FileNotFoundError(f"{name}: no files at {HF_REPO}@{revision}")
    directories = {str(Path(entry.path).parent) for entry in listing}
    if len(directories) != 1:
        raise ValueError(f"{name}: {len(directories)} version directories on the Hub: "
                         f"{sorted(directories)}")
    directory = directories.pop()

    files = []
    for entry in sorted(listing, key=lambda e: e.path):
        local = raw / entry.path
        if not local.exists():
            raise FileNotFoundError(f"{name}: {local} is not on disk")
        lfs = entry.lfs
        lfs_sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) \
            if lfs else None
        digest = file_sha256(local)
        if local.stat().st_size != entry.size:
            raise ValueError(f"{name}: {entry.path} is {local.stat().st_size} bytes on disk, "
                             f"{entry.size} on the Hub")
        if lfs_sha is not None and digest != lfs_sha:
            raise ValueError(f"{name}: {entry.path} sha256 differs from the Hub's LFS record")
        if lfs_sha is None and git_blob_id(local) != entry.blob_id:
            raise ValueError(f"{name}: {entry.path} git blob id differs from the Hub's")
        files.append({
            "name": Path(entry.path).name,
            "path": entry.path,
            "size": int(entry.size),
            "sha256": digest,
            "git_blob_id": entry.blob_id,
            "lfs": lfs_sha is not None,
            "url": f"{HF_HOST}/datasets/{HF_REPO}/resolve/{revision}/{entry.path}",
        })

    def load(suffix):
        match = [f for f in files if f["name"].startswith(suffix)]
        return json.loads((raw / match[0]["path"]).read_text()) if match else {}

    container = load("container_metadata")
    metadata = load("dataset_metadata")
    task = load("task_metadata")
    dtypes = load("dtypes")
    import pyarrow.parquet as pq
    table_file = pq.ParquetFile(raw / directory / DOWNLOAD_FILES["table"])
    parquet = table_file.metadata
    arrow = table_file.schema_arrow
    if set(arrow.names) != set(dtypes):
        raise ValueError(f"{name}: dtypes.json and dataset.parquet name different columns: "
                         f"{sorted(set(arrow.names) ^ set(dtypes))}")
    return {
        "table": name,
        "origin": {
            "host": "huggingface.co",
            "repo_id": HF_REPO,
            "repo_type": "dataset",
            "revision": revision,
            "revision_date": commit.get("date"),
            "revision_title": commit.get("title"),
            "directory": directory,
            "container_uuid": container.get("uuid"),
            "version_comment": container.get("version_comment"),
            "url": f"{HF_HOST}/datasets/{HF_REPO}/tree/{revision}/{directory}",
            "repo_license_tag": repo_license,
        },
        "files": files,
        "dataset": {
            "unique_name": metadata.get("unique_name"),
            "domain": metadata.get("domain_str"),
            "year": metadata.get("dataset_year"),
            "original_source": metadata.get("dataset_source"),
            "original_link": metadata.get("original_dataset_source_download_link"),
            "license": metadata.get("license"),
            "reference_key": metadata.get("academic_reference_bibtex_key"),
            "task_type": task.get("problem_type"),
            "target": task.get("target_column_name"),
            "rows": int(parquet.num_rows),
            "columns": int(parquet.num_columns),
        },
        # The TGB raw schema: columns in file order with the dtype the
        # bundle declares, which is the one the table is read under.
        # ``storage`` is how the parquet file happens to store the column; it
        # is recorded for the reader and never compared.
        "schema": {
            "rows": int(parquet.num_rows),
            "columns": [{"name": column, "dtype": dtypes[column],
                         "storage": str(arrow.field(column).type)}
                        for column in arrow.names],
        },
    }


def write_record(out: Path, entry: dict) -> Path:
    """Write ``entry`` as ``out/<table>/<section>.json``, one file per section.

    ``origin.json`` also carries the table's name, so every folder says what it
    describes without relying on its own name.
    """
    directory = out / entry["table"]
    directory.mkdir(parents=True, exist_ok=True)
    sections = {section: entry[section] for section in SOURCE_SECTIONS}
    sections["origin"] = {"table": entry["table"], **entry["origin"]}
    for section, content in sections.items():
        (directory / SOURCE_SECTIONS[section]).write_text(json.dumps(content, indent=1) + "\n")
    return directory


# --------------------------------------------------------------------------
# Writing the TGB sections
# --------------------------------------------------------------------------

def write_split_indices(path: Path, positions: dict) -> Path:
    """Write a split as one row per raw row: ``row`` (ascending) and ``split``.

    ``positions`` maps each split to the raw row positions it holds.  The file
    is small because consecutive positions delta-encode to almost nothing and
    the split label is a two-bit dictionary code: about 0.15 bytes per row.
    The writer settings are fixed so the same split always gives the same
    bytes with the same pyarrow; what the benchmark pins is the shipped
    file's sha256, not the ability to regenerate it.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = np.concatenate([np.asarray(positions[split], dtype=np.int64) for split in SPLITS])
    labels = np.concatenate([np.full(len(positions[split]), code, dtype=np.int8)
                             for code, split in enumerate(SPLITS)])
    order = np.argsort(rows, kind="stable")
    table = pa.table({
        "row": pa.array(rows[order], pa.int64()),
        "split": pa.DictionaryArray.from_arrays(pa.array(labels[order], pa.int8()),
                                                pa.array(SPLITS, pa.string())),
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=19,
                   use_dictionary=["split"], column_encoding={"row": "DELTA_BINARY_PACKED"},
                   write_statistics=False)
    return Path(path)


def write_tgb_sections(name: str, sections: dict) -> None:
    """Write the TGB sections of ``name`` that ``sections`` holds (maintainers)."""
    directory = source_dir(name)
    for section, file in TGB_SECTIONS.items():
        if section in sections:
            (directory / file).write_text(section_text(sections[section]))


# --------------------------------------------------------------------------
# Fitting and recording the TGB column specification
# --------------------------------------------------------------------------

def _vocabulary(values: pd.Series, max_categories: int) -> tuple[str, ...]:
    """The declared domain of one categorical column.

    ``values`` is the column over every held row, all splits together: the
    schema declares a column's domain and the preprocessor rejects any value
    outside it, so a level that only shows up in the test split still has to
    be declared.

    A column with more than ``max_categories`` labels keeps that many, the
    most frequent over those same rows, and folds the tail into :data:`OTHER`;
    ``OTHER`` and ``MISSING`` are levels on top of the limit.  Frequency is
    counted over the whole table rather than the training split: the
    vocabulary is part of the schema, and must not depend on which rows a
    split holds.  Labels with equal counts are
    ranked by :func:`_text_order`, so which of them survive a tie at the limit does
    not depend on how a library happens to order equal counts.  Zero declares
    everything, and then no ``OTHER`` code exists for the model to emit.
    """
    has_missing = bool(values.isna().any())
    present = values.dropna().astype(str)
    reserved = [MISSING] if has_missing else []
    if max_categories and present.nunique() > max_categories:
        counts = present.value_counts(sort=False)
        ranked = sorted(counts.index, key=lambda label: (-int(counts[label]), _text_order(label)))
        kept = sorted(ranked[:max_categories], key=_text_order)
        return tuple(kept + [OTHER] + reserved)
    return tuple(sorted(set(present), key=_text_order) + reserved)


def timestamp_columns(frame: pd.DataFrame, dtypes: dict) -> tuple:
    """Datetime columns of a whole table that carry a time of day.

    A datetime column is a *timestamp* when any of its values is not at
    midnight, and a *date* otherwise.  A date stays one numerical column of
    days.  A timestamp becomes two, ``<column>__days`` and ``<column>__hours``:
    as one number its time of day is the fraction of a day count spanning
    years, which nothing downstream can use.  ``frame`` is the raw table,
    before any split, so a column is the same kind in every split.
    """
    found = []
    for column, dtype in dtypes.items():
        if kind_of(dtype) != "datetime" or isinstance(frame[column].dtype, pd.PeriodDtype):
            continue
        hours = _hours(frame[column])
        if np.any(hours[np.isfinite(hours)] != 0):
            found.append(column)
    return tuple(found)


def describe(column: str, kind: str, task: dict) -> str:
    """What the benchmark says about a column, which is only its role."""
    if column == task["target_column_name"]:
        return f"the prediction target, {task['problem_type'].replace('_', ' ')}"
    if kind == "datetime":
        return f"{column}, in days since {EPOCH.date()}"
    return ""


def fit_column_spec(name: str, frame, splits: dict, dtypes: dict, task: dict, *,
                    max_categories: int = MAX_CATEGORIES, vocabularies: dict | None = None,
                    drop: dict | None = None, timestamps: tuple = ()) -> dict:
    """Fit a table's column specification: ``{"columns", "column_order", "dropped"}``.

    The fit mode of the benchmark, run by the creator once per table (and
    again after a policy change).  ``frame`` is every held row, all splits
    together, and ``splits`` the splits.  Vocabularies, and the decision to
    add a missing-indicator, are read over every held row, because both are
    part of the schema and a schema that changed between splits would be a
    different table; fills come from the training split alone.  ``drop``
    maps columns to leave out to the reason
    (:data:`tabgen_bench.registry.DROPPED_COLUMNS`), ``timestamps`` names
    the datetime columns that carry a time of day
    (:func:`timestamp_columns`), and ``vocabularies`` holds columns to the
    vocabularies given instead of fitting them, which is how a refit keeps
    the stored ones.  A build never calls this: it applies what this stored
    (:func:`tabgen_bench.preprocessing.encode_splits`).
    """

    # One pass over the raw columns in file order, which is the order the
    # prepared columns keep.
    train = splits["train"]
    columns, dropped = [], {}
    for column, dtype in dtypes.items():
        kind = kind_of(dtype)
        # A listed column is left out, with its reason.
        if drop and column in drop:
            dropped[column] = drop[column]
            continue
        # A categorical column: its vocabulary, fitted over every held row.
        if kind == "categorical":
            labels = _convert_to_string_labels(frame[column])
            # A constant column is kept: one label, and a generator that
            # emits anything else is wrong about it.  An all-null column is
            # the one-level vocabulary ``(missing)``.
            categories = (vocabularies[column] if vocabularies and column in vocabularies
                          else _vocabulary(labels, max_categories))
            columns.append({"name": column, "kind": "categorical",
                            "categories": list(categories),
                            "description": describe(column, kind, task)})
            continue

        # A numerical or datetime column, read as numbers (days for a date).
        convert = _days if kind == "datetime" else _numbers
        values = convert(frame[column])
        finite = values[np.isfinite(values)]
        # No value at all: the one-level categorical ``(missing)``.
        if len(finite) == 0:
            # Null in every row: there is no value to fill it with, so it is
            # kept the way a categorical null is, as the one level ``(missing)``.
            columns.append({"name": column, "kind": "categorical", "source": kind,
                            "all_null": True, "categories": [MISSING],
                            "description": f"{describe(column, kind, task)}; missing in every row"})
            continue
        # A timestamp: two numerical columns, days and hours, each with its
        # own fill from the training split.
        if column in timestamps:
            role = describe(column, kind, task)
            for part, source, read in TIMESTAMP_PARTS:
                part_name = f"{column}__{part}"
                part_values = read(frame[column])
                part_finite = part_values[np.isfinite(part_values)]
                part_train = read(train[column])
                part_train = part_train[np.isfinite(part_train)]
                fill = float(np.median(part_train)) if len(part_train) else float(part_finite[0])
                what = (f"date of {column}, in whole days since {EPOCH.date()}" if part == "days"
                        else f"time of day of {column}, in hours since midnight")
                columns.append({"name": part_name, "kind": "numerical", "source": source,
                                "part_of": column, "fill": fill,
                                "description": what if kind_of(dtype) != "datetime" or
                                role.startswith(f"{column}, in days") else f"{what}; {role}"})
        # Any other number: one column, filled with its training median (a
        # constant column keeps its constant).
        else:
            train_values = convert(train[column])
            train_finite = train_values[np.isfinite(train_values)]
            median = float(np.median(train_finite)) if len(train_finite) else float(finite[0])
            columns.append({"name": column, "kind": "numerical", "source": kind,
                            "fill": median, "description": describe(column, kind, task)})
        # An imputed value is indistinguishable from a measured one, so when
        # a column is missing often enough to matter the fact of its absence
        # gets its own column.
        missing_frac = float(np.mean(~np.isfinite(values)))
        if missing_frac >= MISSING_INDICATOR_MIN_FRAC:
            indicator = f"{column}__missing"
            columns.append({
                "name": indicator, "kind": "categorical", "indicator_of": column,
                "categories": ["present", "missing"],
                "description": f"whether {column} was recorded",
            })

    if not columns:
        raise ValueError(f"{name}: every column was dropped")
    return {"columns": columns, "column_order": [column["name"] for column in columns],
            "dropped": dropped}


def record_columns(name: str, data_root: Path = DATA_ROOT, *,
                   refit_vocabularies: bool = False) -> dict:
    """Fit ``name``'s column specification and store it and its digests in the package.

    The creator's step after :func:`draw_split`, and again after a policy
    change.  The raw table and its stored split are loaded and checked as a
    build loads them (:func:`tabgen_bench.preprocessing.load_raw_splits`);
    the specification is fitted (:func:`fit_column_spec`) under the
    benchmark's policy, with the columns :data:`registry.DROPPED_COLUMNS`
    names left out; the splits are encoded and digested; and
    ``tgb_columns.json`` and ``tgb_prepared.json`` are written.  When the
    package already holds a specification for the table its vocabularies
    are kept, so a policy change reaches only what it concerns, unless
    ``refit_vocabularies`` asks for them to be fitted again (as a change of
    the vocabulary limit needs).  The table is then built with
    :func:`tabgen_bench.preprocessing.prepare_table`, which applies what was
    just stored and must reproduce the digests: the fit and the apply agree.
    Returns the build's spec.
    """

    loaded = load_raw_splits(name, data_root)
    frame, dtypes, task, splits = (loaded[k] for k in ("frame", "dtypes", "task", "splits"))
    # The columns the benchmark leaves out, by the registry's list.
    drop = {}
    for column, reason in dropped_columns(name).items():
        if column not in dtypes:
            raise ValueError(f"{name}: DROPPED_COLUMNS names {column!r}, "
                             "which the table does not have")
        if column == task["target_column_name"]:
            raise ValueError(f"{name}: DROPPED_COLUMNS names the target {column!r}")
        drop[column] = reason
    # The stored vocabularies survive a refit unless asked otherwise.
    vocabularies = None
    stored = read_source(name) or {}
    if "columns" in stored and not refit_vocabularies:
        vocabularies = {column["name"]: list(column["categories"])
                        for column in stored["columns"]["columns"]
                        if column["kind"] == "categorical" and "indicator_of" not in column}
    held = pd.concat(splits.values(), ignore_index=True)
    spec = fit_column_spec(name, held, splits, dtypes, task, vocabularies=vocabularies,
                           drop=drop, timestamps=timestamp_columns(frame, dtypes))
    encoded = encode_splits(spec, splits)
    digests = {s: compute_dataframe_digest(f) for s, f in encoded.items()}
    record_tgb_columns(name, spec, digests, {s: int(len(f)) for s, f in encoded.items()})
    return prepare_table(name, data_root)


def record_tgb_columns(name: str, spec: dict, digests: dict, rows: dict) -> None:
    """Write ``tgb_columns.json`` and ``tgb_prepared.json`` for ``name``.

    ``spec`` is a fitted column specification, ``digests`` the content
    digest of each encoded split and ``rows`` their row counts.  The split
    is never touched here: :func:`draw_split` is what stores one.
    """
    import platform

    import pyarrow

    columns = {"table": name, "column_order": spec["column_order"],
               "columns": spec["columns"], "dropped": spec["dropped"]}
    prepared = {
        "table": name,
        "columns_sha256": hashlib.sha256(section_text(columns).encode()).hexdigest(),
        "digest_label": DIGEST_LABEL,
        "digests": digests,
        "rows": rows,
        "columns": len(spec["column_order"]),
        "policy": {"max_categories": MAX_CATEGORIES,
                   "missing_indicator_min_frac": MISSING_INDICATOR_MIN_FRAC},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "tabgen_bench_version": __version__,
    }
    write_tgb_sections(name, {"columns": columns, "prepared": prepared})


# --------------------------------------------------------------------------
# Drawing the TGB split
# --------------------------------------------------------------------------

# The benchmark's split: a random 80/10/10 cut of every raw row (or of the
# rows of the table's row subset), the same rule for every table, drawn once
# with ``np.random.default_rng(crc32(name) ^ seed)`` and then stored.
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SPLIT_SEED = 42


def validation_split(train, fraction: float = VAL_FRACTION, *,
                     rng: np.random.Generator):
    """Cut a random ``fraction`` of ``train``'s rows out as a validation set.

    Returns ``(train_positions, validation_positions)`` into ``train``, a
    partition of its rows, each ascending.  At least one row stays on each
    side.
    """
    n = len(train)
    if n < 2:
        raise ValueError("a training fold needs at least two rows to cut")
    n_val = min(max(1, int(round(n * fraction))), n - 1)
    order = rng.permutation(n)
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def fraction_splits(frame, *, val_fraction: float, test_fraction: float,
                    rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Cut the whole table into train / validation / test at random.

    An alternative to the benchmark fold: ``test_fraction`` of the rows are
    the test split and ``val_fraction`` the validation split, both measured
    against the whole table.  Returns positions into ``frame``, a partition
    of its rows.
    """
    if not 0 < test_fraction < 1 or not 0 <= val_fraction < 1 - test_fraction:
        raise ValueError("fractions must satisfy 0 < test < 1 and "
                         "0 <= val < 1 - test")
    rest, test = validation_split(frame, test_fraction, rng=rng)
    if val_fraction == 0:
        return {"train": rest, "validation": np.array([], dtype=int), "test": test}
    remaining = frame.iloc[rest].reset_index(drop=True)
    train, val = validation_split(remaining, val_fraction / (1 - test_fraction), rng=rng)
    return {"train": rest[train], "validation": rest[val], "test": test}


def _table_rng(name: str, seed: int) -> np.random.Generator:
    """crc32 rather than ``hash()``: Python salts string hashing per process."""
    return np.random.default_rng(zlib.crc32(name.encode()) ^ (seed & 0xFFFFFFFF))


def subset_rows(frame, name: str, rule: dict) -> tuple:
    """Positions of the rows of ``rule["groups"]`` whole groups of one column.

    The groups are the distinct labels of ``rule["column"]``, every one of
    which must be present (a null is a row with no group, and is refused).
    They are sorted by :func:`_text_order`, permuted by ``_table_rng`` seeded on
    ``"<table>/<column>"`` and the first ``groups`` kept; the returned
    positions are every row of a kept group, in file order.  A column with
    no more groups than asked keeps every row.  The second value is the
    record of the draw.
    """
    column, groups, seed = rule["column"], int(rule["groups"]), int(rule["seed"])
    if column not in frame.columns:
        raise ValueError(f"{name}: ROW_SUBSETS names {column!r}, which the table does not have")

    labels = _convert_to_string_labels(frame[column])
    if labels.isna().any():
        raise ValueError(f"{name}: {column!r} has nulls; a row subset needs a group for every row")
    distinct = sorted(set(labels.astype(str)), key=_text_order)
    if groups <= 0:
        raise ValueError(f"{name}: ROW_SUBSETS asks for {groups} groups")
    rng = _table_rng(f"{name}/{column}", seed)
    kept = set(np.asarray(distinct, dtype=object)[rng.permutation(len(distinct))][:groups])
    positions = np.flatnonzero(labels.astype(str).isin(kept).to_numpy())
    record = {"column": column, "groups": groups, "groups_total": len(distinct),
              "groups_kept": len(kept), "rows_kept": int(len(positions)),
              "rows_total": int(len(frame)), "seed": seed,
              "rng": "numpy default_rng(crc32('<table>/<column>') ^ seed); see _table_rng"}
    return positions, record


def draw_split(name: str, data_root: Path = DATA_ROOT, *,
               val_fraction: float = VAL_FRACTION, test_fraction: float = TEST_FRACTION,
               seed: int = SPLIT_SEED) -> dict:
    """Draw ``name``'s split and store it in the package as the TGB one.

    The raw table is read from ``<data_root>/raw``; if the registry holds a
    row subset for the table (:data:`registry.ROW_SUBSETS`) the subset is
    drawn first (:func:`subset_rows`) and the split is cut on its rows.
    ``tgb_split_indices.parquet`` and ``tgb_split.json`` are written;
    returns the split record.  A table's column specification and digests
    are then recorded by :func:`record_columns`.
    """
    import numpy as np

    from .datasets import read_raw
    from .registry import row_subset

    raw = Path(data_root) / "raw"
    frame, *_ = read_raw(name, raw)
    rule = row_subset(name)
    subset, subset_record = (subset_rows(frame, name, rule) if rule else (None, None))
    sub = np.arange(len(frame)) if subset is None else subset
    drawn_on = frame if subset is None else frame.iloc[sub].reset_index(drop=True)
    rng = _table_rng(name, seed)
    drawn = fraction_splits(drawn_on, val_fraction=val_fraction, test_fraction=test_fraction,
                            rng=rng)
    positions = {s: np.sort(sub[np.asarray(idx, dtype=np.int64)]) for s, idx in drawn.items()}
    check_partition(name, positions, len(frame),
                    held_rows=subset_record["rows_kept"] if subset_record else None)
    path = write_split_indices(split_indices_path(name), positions)
    table_file = dataset_dir(name, raw) / DOWNLOAD_FILES["table"]
    record = {
        "table": name,
        "file": TGB_SPLIT_INDICES,
        "sha256": file_sha256(path),
        "raw_file": str(table_file.relative_to(raw)),
        "raw_sha256": file_sha256(table_file),
        "raw_rows": int(len(frame)),
        "rows": {s: int(len(positions[s])) for s in SPLITS},
        **({"row_subset": subset_record} if subset_record else {}),
        # How the split was drawn.  Kept for the record: the stored positions,
        # not this rule, are what defines the split.
        "rule": {"drawn_as": "fractions", "val_fraction": val_fraction,
                 "test_fraction": test_fraction, "seed": seed,
                 "rng": "numpy default_rng(crc32(name) ^ seed); see _table_rng"},
    }
    write_tgb_sections(name, {"split": record})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=DATA_ROOT / "raw",
                        help="the downloaded bundle (default: <data root>/raw)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="tables to record (default: every table of TABLES); a "
                             "new one joins the benchmark once it is also listed there")
    parser.add_argument("--revision", default=None,
                        help="commit to pin (default: the one the download cache "
                             "recorded, else the repository's current head)")
    parser.add_argument("--out", type=Path, default=SOURCES)
    parser.add_argument("--dropped-columns", action="store_true",
                        help="instead of recording: compute the columns the cardinality "
                             "rule drops, from the raw tables and the TGB training "
                             "splits, and print them as registry.DROPPED_COLUMNS")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="with --dropped-columns or --draw-split: the data root "
                             "whose raw/ to read")
    parser.add_argument("--draw-split", action="store_true",
                        help="instead of recording: draw the TGB split of each of "
                             "--datasets from its raw rows (and its row subset, if the "
                             "registry holds one) and store it in the package; then "
                             "--record-columns records the columns and digests")
    parser.add_argument("--record-columns", action="store_true",
                        help="instead of recording: fit the column specification of each "
                             "of --datasets (default: every table) on its stored split, "
                             "store it with the content digests, and build the table")
    parser.add_argument("--refit-vocabularies", action="store_true",
                        help="with --record-columns: fit the vocabularies again instead "
                             "of keeping the stored ones, as a change of the vocabulary "
                             "limit needs")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    if args.record_columns:
        for name in args.datasets or TABLES:
            spec = record_columns(name, args.data_root, refit_vocabularies=args.refit_vocabularies)
            print(f"{name}: columns and digests stored; {spec['rows']}, "
                  f"{len(spec['column_order'])} columns, dropped {len(spec['dropped'])}",
                  flush=True)
        return
    if args.draw_split:
        if not args.datasets:
            raise SystemExit("--draw-split needs --datasets: a split is drawn once per table")
        for name in args.datasets:
            record = draw_split(name, args.data_root, val_fraction=args.val_fraction,
                                test_fraction=args.test_fraction, seed=args.seed)
            print(f"{name}: split stored, {record['rows']}"
                  + (f", row subset {record['row_subset']['rows_kept']} of "
                     f"{record['raw_rows']} rows" if record.get("row_subset") else ""),
                  flush=True)
        return

    if args.dropped_columns:
        found = find_dropped_columns(args.datasets, args.data_root)
        print(dropped_columns_literal(found))
        same = all(found.get(t) == DROPPED_COLUMNS.get(t) for t in (args.datasets or TABLES))
        print("# matches registry.DROPPED_COLUMNS" if same
              else "# DIFFERS from registry.DROPPED_COLUMNS: update the registry")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    names = args.datasets or list(TABLES)
    raw = args.raw.resolve()

    revision = args.revision
    if revision is None:
        cached = {cached_revision(name, raw) for name in names}
        if len(cached) == 1 and None not in cached:
            revision = cached.pop()
        else:
            revision = api.dataset_info(HF_REPO).sha
            print(f"download cache names no single commit ({cached}); pinning the head")
    info = api.dataset_info(HF_REPO, revision=revision)
    if info.sha != revision:
        raise ValueError(f"{revision} resolves to {info.sha}; pass a full commit id")
    commits = {c.commit_id: {"date": c.created_at.isoformat(), "title": c.title}
               for c in api.list_repo_commits(HF_REPO, repo_type="dataset")}
    card = info.card_data
    repo_license = (card.get("license") if hasattr(card, "get") else getattr(card, "license", None)) \
        if card is not None else None
    head = api.dataset_info(HF_REPO).sha
    print(f"{HF_REPO} @ {revision} ({commits.get(revision, {}).get('date')}); "
          f"head is {'the same commit' if head == revision else head}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        entry = record(name, raw, api, revision, commits.get(revision, {}), repo_license)
        write_record(args.out, entry)
        total = sum(f["size"] for f in entry["files"])
        print(f"  {name}: {len(entry['files'])} files, {total / 1e6:.1f} MB, "
              f"version {entry['origin']['container_uuid']}")
    print(f"{len(names)} records written to {args.out}")
    unlisted = sorted(set(recorded_tables(args.out)) - set(TABLES))
    if unlisted:
        print(f"recorded but not listed in registry.TABLES, so not part of the "
              f"benchmark: {unlisted}")


if __name__ == "__main__":
    main()
