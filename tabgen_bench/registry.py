"""The benchmark's registry: which tables it has and where each comes from.

The data origin is the BeyondArena bundle on Hugging Face
(``TabArena/BeyondArena``, a dataset repository).  A repository can change, so
the benchmark ships a folder ``table_sources/<table>/`` for every table,
pinning exactly which bytes it is built from.  The benchmark's tables are
listed explicitly in :data:`TABLES` below; every other module works on
exactly those (:func:`benchmark_tables`), and each must have a folder.  Each
folder holds one JSON per concern:

* ``origin.json``: the repository and the commit (``revision``) with its
  date and title; the dataset's version directory in the bundle
  (``<table>/<container uuid>``), its container uuid and version comment; the
  URL of that directory at the commit; and the table's name.  The bundle's
  own container checksum is not copied: nothing uses it, and the
  ``container_metadata.json`` that holds it is one of the files
  ``files.json`` pins by sha256.
* ``files.json``: the six files of that directory -- size, sha256, the git
  blob id the Hub lists, whether it is an LFS file, and a download URL pinned
  to the commit.
* ``dataset.json``: what the bundle's metadata says about it -- original
  source and link, licence, year, citation key, task, target, raw shape.
* ``schema.json``: the TGB raw schema -- the raw table's row count and
  its columns in file order, each with the dtype BeyondArena declares for it.
* ``tgb_split_indices.parquet`` and ``tgb_split.json``: the split itself.  The first
  holds one row per raw row -- its position in ``dataset.parquet`` and the
  split it belongs to -- so the benchmark's split is stored data, not the
  output of a random generator.  The second records that file's sha256, the
  sha256 of the raw table the positions refer to, the row counts, and how the
  split was originally drawn.
* ``tgb_columns.json``: the TGB column specification -- the prepared
  columns in order, each categorical column's vocabulary, each numerical
  column's fill value and whether it has a missing-indicator, and the raw
  columns that are dropped.  A TGB build applies it; it fits nothing.
* ``tgb_prepared.json``: the sha256 of ``tgb_columns.json`` and a content digest of
  each prepared split, so a user can confirm that preprocessing on their
  machine produced the TGB data.

Together these are the reproducibility chain every TGB build follows
(:func:`tabgen_bench.preprocessing.prepare_table`)::

    TGB raw schema      schema.json
            |
    raw-data checksum         files.json
            |
    stored row indices        tgb_split_indices.parquet
            |
    split-file checksum       tgb_split.json
            |
    train / validation / test
            |
    stored column spec        tgb_columns.json, behind its sha256 in tgb_prepared.json
            |
    content digests           tgb_prepared.json

This module reads those records (:func:`read_source`).  Writing them is the
maintainer's tool, :mod:`tabgen_bench.creator_tool`: a record is
cross-checked against the Hub at the pinned commit, never copied.

It is also the package's base module: the shared locations and names
(``DATA_ROOT``, ``HF_REPO``, ``DOWNLOAD_FILES``, ``SPLITS``) and the two content
hashes the records are made of live here, and it imports nothing else from
the package.

The records ship inside the package (``tabgen_bench/table_sources/``).  Data
is never written there: it goes to the data root, which is the
``TABGEN_BENCH_DATA`` environment variable when set and
``~/.cache/tabgen_bench`` otherwise, and which every function and command
also takes explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import __version__  # noqa: F401  (re-exported for the modules below)

PACKAGE_ROOT = Path(__file__).resolve().parent

# Where downloaded and prepared data lives.
DATA_ENV = "TABGEN_BENCH_DATA"


def default_data_root() -> Path:
    """``$TABGEN_BENCH_DATA`` when set, ``~/.cache/tabgen_bench`` otherwise."""
    configured = os.environ.get(DATA_ENV)
    default_path = Path.home() / ".cache" / "tabgen_bench"
    return Path(configured).expanduser() if configured else default_path


# TabGenBench data root
DATA_ROOT = default_data_root()

HF_REPO = "TabArena/BeyondArena"
HF_HOST = "https://huggingface.co"
HF_URL = f"{HF_HOST}/datasets/{HF_REPO}"

# The six artifacts a BeyondArena dataset directory holds.
# The text embedding is never downloaded.
DOWNLOAD_FILES = {
    "table": "dataset.parquet",
    "dtypes": "dtypes.json",
    "dataset": "dataset_metadata.dataset-mold-v1.json",
    "task": "task_metadata.predictive-ml-task-mold-v1.json",
    "splits": "experiment_metadata.predictive-ml-splits-mold-v1.json",
    "container": "container_metadata.json",
}

# TabGenBench splits
SPLITS = ("train", "validation", "test")

# One folder per benchmark table, shipped inside the package.  Functions read
# this name when they are called, so a test can point it at another folder.
SOURCES = PACKAGE_ROOT / "table_sources"
# The origin of a table, written by :func:`tabgen_bench.creator_tool.record`:
# one JSON per section.  A table with these is downloaded at the pinned commit
# and checked against them before it is used.
# Each section is a key of the record (``read_source(name)[section]``) and
# the file that holds it.
SOURCE_SECTIONS = {
    "origin": "origin.json",
    "files": "files.json",
    "dataset": "dataset.json",
    "schema": "schema.json",
}
# The TGB split, the TGB column specification and the digests of
# the prepared output, written by
# ``creator_tool.draw_split`` and ``creator_tool.record_columns``.  The ``tgb_`` prefix
# shows at a glance which files describe the origin and which the benchmark's
# own build; the stored split sits beside them.
TGB_SECTIONS = {
    "split": "tgb_split.json",
    "columns": "tgb_columns.json",
    "prepared": "tgb_prepared.json",
}
TGB_SPLIT_INDICES = "tgb_split_indices.parquet"

# The tables in use: the benchmark is exactly these BeyondArena datasets, by
# the bundle's own directory names.  Each must have an origin record under
# ``table_sources/``.  To add a table, record it
# (``python -m tabgen_bench.creator_tool --datasets <name>``) and list it here;
# a recorded folder that is not listed is not part of the benchmark.
TABLES = (
    "5g_energy_consumption",
    "acquire_valued_shoppers_challenge",
    "allstate_claims_severity",
    "amex_non_iid",
    "aps_failure",
    "climate_model_weather_forecasting",
    "consumer_complaints",
    "cooking_time",
    "covertype",
    "customer_satisfaction_in_airline",
    "delivery_eta",
    "diabetes_130_us",
    "electric_motor_temperature_prediction",
    "give_me_some_credit",
    "homesite_quote_conversion",
    "hotel_booking_demand",
    "kick",
    "kickstarter",
    "lending_club",
    "mercari_price_suggestion",
    "porto_seguro",
    "rossmann_store_sales",
    "sdss_17",
    "sepsis_prediction",
    "sepsis_survival_minimal_clinical_records",
    "sf_permit_time",
    "video_transcoding_time_prediction",
    "wids_diabetes_mellitus",
)


# Columns the benchmark reads under another dtype than BeyondArena declares.
# The TGB raw schema still records the bundle's own declaration; this is
# applied on top of it when a raw table is read, before anything else looks at
# the column.  Each entry is a judgement about one column and says why.
DTYPE_OVERRIDES = {
    "give_me_some_credit": {
        # How much of their revolving credit (credit cards and personal lines
        # of credit, not mortgages or instalment loans) a borrower is using:
        # the total balance divided by the total credit limit, from the
        # Kaggle "Give Me Some Credit" competition.  A continuous ratio,
        # 0.0 to 1.0 for 98% of borrowers (median 0.15) and above 1 for the
        # rest, with 125,728 distinct values in 150,000 rows, that the
        # bundle declares ``category``.  As a category it is one label per
        # row; it is a number.
        "RevolvingUtilizationOfUnsecuredLines": "float64",
    },
}


# Columns the benchmark leaves out, with the reason recorded beside each.  Free
# text and identifiers carry about one label per row, which no vocabulary can
# hold and no model can learn from.  The list was made by the creator's tool
# (``python -m tabgen_bench.creator_tool --dropped-columns``): a categorical
# column, the target excepted, whose distinct labels in the TGB training split
# exceed a tenth of that split's rows.  It is data, not a rule a build applies:
# a build drops exactly these columns, whatever rows it holds.
DROPPED_COLUMNS = {
    "amex_non_iid": {
        "customer_ID":
            "high cardinality: 103402 labels in 999684 training rows, ratio 0.103 above 0.1",
    },
    "consumer_complaints": {
        "Consumer complaint narrative":
            "high cardinality: 700633 labels in 980912 training rows, ratio 0.714 above 0.1",
    },
    "kickstarter": {
        "blurb":
            "high cardinality: 148032 labels in 149694 training rows, ratio 0.989 above 0.1",
        "name":
            "high cardinality: 149308 labels in 149694 training rows, ratio 0.997 above 0.1",
        "profile_blurb":
            "high cardinality: 27006 labels in 149694 training rows, ratio 0.180 above 0.1",
        "creator_name":
            "high cardinality: 121278 labels in 149694 training rows, ratio 0.810 above 0.1",
    },
    "lending_club": {
        "desc":
            "high cardinality: 88091 labels in 851801 training rows, ratio 0.103 above 0.1",
        "emp_title":
            "high cardinality: 275540 labels in 851801 training rows, ratio 0.323 above 0.1",
    },
    "mercari_price_suggestion": {
        "name":
            "high cardinality: 847792 labels in 1000000 training rows, ratio 0.848 above 0.1",
        "item_description":
            "high cardinality: 872232 labels in 1000000 training rows, ratio 0.872 above 0.1",
    },
    "sf_permit_time": {
        "Description":
            "high cardinality: 90866 labels in 93564 training rows, ratio 0.971 above 0.1",
    },
}


def dropped_columns(name: str) -> dict:
    """``{column: reason}`` the benchmark leaves out of ``name``."""
    return dict(DROPPED_COLUMNS.get(name, {}))


# Tables the benchmark holds a subset of, by whole groups of one column.  The
# rule is applied to the raw table before it is split; the rows of the groups
# not drawn are in no split.  ``seed`` feeds ``preprocessing._table_rng`` on
# ``"<table>/<column>"``, so the draw is independent of the split's.
ROW_SUBSETS = {
    "sepsis_prediction": {
        # One row per ICU hour, 31,930 patients, 8 to 336 rows each (median
        # 39).  Drawing 16,000 patients halves the table and, under the
        # 15,000-label limit, leaves 1,000 patients' rows in ``(other)``
        # rather than a third of them.
        "column": "Patient_ID",
        "groups": 16_000,
        "seed": 42,
    },
}


def row_subset(name: str) -> dict | None:
    """The whole-group row subset the benchmark holds of ``name``, or ``None``."""
    rule = ROW_SUBSETS.get(name)
    return dict(rule) if rule else None


def dtype_overrides(name: str) -> dict:
    """``{column: dtype}`` the benchmark reads ``name`` with instead of the declared dtype."""
    return dict(DTYPE_OVERRIDES.get(name, {}))


# --------------------------------------------------------------------------
# Reading the registry
# --------------------------------------------------------------------------

def benchmark_tables() -> tuple[str, ...]:
    """The benchmark's tables: :data:`TABLES`, each checked to have a record.

    A listed table without ``table_sources/<table>/origin.json`` is an error
    naming it, so the list and the records cannot drift apart silently.
    """
    missing = [name for name in TABLES if not (source_dir(name) / "origin.json").exists()]
    if missing:
        raise FileNotFoundError(
            f"listed in registry.TABLES but without a record under {SOURCES}: {missing}; "
            f"record them with python -m tabgen_bench.creator_tool --datasets {' '.join(missing)}")
    return TABLES


def prepared_dir(data_root: Path, name: str) -> Path:
    """Where the prepared splits of ``name`` live under a data root."""
    return Path(data_root) / "prepared" / name


def source_dir(name: str) -> Path:
    """The record folder of ``name`` (``table_sources/<name>``)."""
    return SOURCES / name


def read_source(name: str) -> dict | None:
    """The record of ``name``, or None when the benchmark has none.

    The origin sections are always present; ``split`` and ``prepared`` are
    present for a table that has a TGB split.
    """
    directory = source_dir(name)
    if not (directory / "origin.json").exists():
        return None
    source = {section: json.loads((directory / file).read_text())
              for section, file in SOURCE_SECTIONS.items()}
    for section, file in TGB_SECTIONS.items():
        path = directory / file
        if path.exists():
            source[section] = json.loads(path.read_text())
    return source


def table_info(name: str) -> dict:
    """What the benchmark records about ``name``, for a reader.

    Returns the table's task, target, raw shape, original source, licence
    label and link, the pinned repository and commit, and -- when the table
    has a TGB split -- the rows per split.  Raises ``KeyError`` for a
    table the benchmark does not have.
    """
    source = read_source(name)
    if source is None:
        raise KeyError(f"{name!r} is not a benchmark table; see tabgen_bench.list_tables()")
    info = {"table": name, **source["dataset"],
            "repo_id": source["origin"]["repo_id"],
            "revision": source["origin"]["revision"],
            "url": source["origin"]["url"]}
    if "split" in source:
        info["split_rows"] = dict(source["split"]["rows"])
    return info


def list_tables() -> tuple[str, ...]:
    """The names of the benchmark's tables."""
    return benchmark_tables()


# --------------------------------------------------------------------------
# What a record is made of: content hashes and the download cache's commit
# --------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path: Path) -> str:
    """The id git gives a file's content: sha1 over ``blob <size>\\0`` + bytes."""
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_revision(name: str, raw: Path) -> str | None:
    """The bundle commit the download cache recorded for ``name``'s table file."""
    cache = Path(raw) / ".cache" / "huggingface" / "download" / name
    for meta in sorted(cache.rglob(DOWNLOAD_FILES["table"] + ".metadata")):
        lines = meta.read_text().splitlines()
        if lines:
            return lines[0].strip()
    return None


# --------------------------------------------------------------------------
# The stored split
# --------------------------------------------------------------------------

def split_indices_path(name: str) -> Path:
    """Where the stored split of ``name`` lives."""
    return source_dir(name) / TGB_SPLIT_INDICES


def read_split_indices_file(path: Path) -> dict:
    """Read a stored split back as ``{split: ascending raw row positions}``.

    This only parses the file.  :func:`tabgen_bench.validation.load_split_indices`
    is the checked way in: it verifies the file's sha256 and that the positions
    partition the raw table before handing them out.
    """
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["row", "split"])
    rows = table.column("row").to_numpy()
    labels = np.asarray(table.column("split").cast("string").to_pylist(), dtype=object)
    unknown = sorted(set(labels.tolist()) - set(SPLITS))
    if unknown:
        raise ValueError(f"{path}: unknown split labels {unknown}")
    return {split: rows[labels == split].astype(np.int64) for split in SPLITS}


def section_text(content) -> str:
    """The exact text a record section is stored as; its sha256 is taken of this."""
    return json.dumps(content, indent=1) + "\n"
