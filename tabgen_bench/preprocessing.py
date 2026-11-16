"""Preprocessing: cut a raw table into splits and write the prepared corpus.

:func:`prepare` turns downloaded tables into ``<data root>/prepared/<table>/``
-- three parquet splits and ``spec.json`` -- and writes
``prepared/manifest.json``.  It downloads what is missing.

    # the benchmark
    python -m tabgen_bench.preprocessing

**With no options the build is TGB**, and every table goes through the
reproducibility chain (:func:`prepare_table`): the raw table must have the
TGB raw schema and the recorded sha256; the split is the row positions
stored in the package, taken only after that file's own sha256 and its
partition of the raw rows have been checked; the splits are encoded with the
column specification stored in the package, also behind its sha256; and the
content digest of each prepared split must equal the one the benchmark ships.
Nothing in that path draws a random number or fits a vocabulary, so neither
the split nor the columns can drift with a library version.

The split is always the one the package ships, ``tgb_split_indices.parquet``,
and the columns are always the specification it ships, ``tgb_columns.json``:
a build never draws a split and never fits a column.  Both are the creator's
job (:mod:`tabgen_bench.creator_tool`: ``--draw-split`` and
``--record-columns``), and what the creator makes is stored in the package
before any build uses it.  The only build option is ``--no-strict``, which
writes a table whose digests differ from the benchmark's instead of stopping.

Column policy, applied uniformly:

* A date becomes one numerical column, days since :data:`EPOCH`; the one
  ``period`` column is a date.  A timestamp -- a datetime column in which
  some value has a time of day, judged on the whole table before the split --
  becomes two numerical columns: ``<column>__days``, the whole days since
  :data:`EPOCH`, and ``<column>__hours``, the hours since midnight.
* Booleans and categories are categoricals, rendered as labels; integer-valued
  floats render without a decimal point, so a column declared ``category``
  and stored as ``1.0`` / ``2.0`` gets the labels ``"1"`` / ``"2"``.
* A categorical's vocabulary is every label that occurs in the table, pooled
  over the splits -- a level that only occurs in the test split is still
  declared -- with ``(missing)`` added when the column has nulls.  A column
  with more than ``max_categories`` (15,000) labels keeps its 15,000 most
  frequent labels, counted over the whole table, and folds the rest into
  ``(other)``.  Free-text
  columns not in ``DROPPED_COLUMNS`` are categoricals like any other.
* Before anything else, a few columns are recast to another dtype than
  the bundle declares (:data:`tabgen_bench.registry.DTYPE_OVERRIDES`,
  :func:`apply_dtype_overrides`): a continuous ratio declared ``category``
  is prepared as the number it is.  ``load_raw`` still gives the table as
  declared.
* The columns listed in :data:`tabgen_bench.registry.DROPPED_COLUMNS` are
  left out and listed in the spec with their reason: free text and
  identifiers at about one label per row, found by the creator's tool
  (:func:`tabgen_bench.creator_tool.find_dropped_columns`).
* ``+-inf`` is missing.  A numerical column missing at least
  :data:`MISSING_INDICATOR_MIN_FRAC` (1%) of the time gets a
  ``<column>__missing`` indicator beside it and is filled with its training
  median.  No row is ever dropped for a null.
* No column is dropped for its content: a constant column is kept, and an all-null
  column becomes a categorical column with the single level ``(missing)``.

Each table is written once; a second run skips what is there unless
``--force``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import (
    DATA_ROOT,
    DOWNLOAD_FILES,
    HF_REPO,
    HF_URL,
    TGB_SPLIT_INDICES,
    SPLITS,
    benchmark_tables,
    default_data_root,
    dropped_columns,
    dtype_overrides,
    row_subset,
    read_source,
)
from .validation import (
    DIGEST_LABEL,
    PreparedMismatchError,
    SplitIndicesError,
    compute_dataframe_digest,
    load_column_spec,
    load_split_indices,
    verify_prepared_dataframes,
    verify_source,
)
from .datasets import (
    apply_dtypes,
    dataset_dir,
    kind_of,
    read_raw,
    spec_file,
    split_file,
    table_dir,
)
from .download import download, raw_present

# Days are counted from the Unix epoch; a date before it is simply negative.
EPOCH = pd.Timestamp("1970-01-01")

# The labels of a folded tail and of an absent value.
MISSING = "(missing)"
OTHER = "(other)"

# The most labels a categorical column keeps.  A column with more keeps its
# most frequent labels, counted over the whole table, and the rest fold into ``(other)``, which, like
# ``(missing)``, is a level on top of the limit.  0 is no limit.  The limit is
# a bound on what a model has to hold: a method that scores every label of a
# column needs memory in proportion to the widest vocabulary, and with no
# limit neither TabDiff nor a shared-index denoiser fits one training step of
# ``cooking_time`` or ``lending_club`` on an 80 GB card.  Five columns in
# three tables exceed it.  Columns that are close to one label per row are
# left out by ``registry.DROPPED_COLUMNS`` before this is applied.
MAX_CATEGORIES = 15_000
# A numerical column absent at least this often gets its own indicator column.
MISSING_INDICATOR_MIN_FRAC = 0.01


# --------------------------------------------------------------------------
# Column values
# --------------------------------------------------------------------------

def _text_order(value: str):
    """Sort numeric-looking labels as numbers, everything else as text.

    A total order: labels that are the same number (``"01"`` and ``"001"``)
    are ranked by their text, so a sort by this key has one outcome.
    """
    try:
        number = float(value)
    except ValueError:
        return (1, 0.0, value)
    if number != number:                      # "nan" is text, not a number to compare
        return (1, 0.0, value)
    return (0, number, value)


def _convert_to_string_labels(series: pd.Series) -> pd.Series:
    """Render a column as category labels, keeping nulls as ``NaN``.

    Integer-valued numerics render without a decimal point, so a column the
    benchmark declares ``category`` and the parquet stores as ``2.0`` gets
    the label ``"2"`` -- and the same label whether or not a null elsewhere
    in the column forced it to float.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        # Read the values' own type: a ``category`` column the parquet
        # stored as float carries float categories, and ``1.0`` has to label
        # as ``"1"`` here as it would have as an int64 column.
        series = series.astype(object)
    inferred = pd.api.types.infer_dtype(series, skipna=True)
    if pd.api.types.is_bool_dtype(series) or inferred == "boolean":
        return series.map({True: "True", False: "False"}).astype(object)
    if pd.api.types.is_numeric_dtype(series) or inferred in (
            "integer", "floating", "mixed-integer-float", "decimal"):
        values = pd.to_numeric(series, errors="coerce")
        finite = values.dropna()
        if len(finite) and np.all(np.equal(np.mod(finite.to_numpy(float), 1), 0)):
            values = values.astype("Int64")
        return values.astype("string").astype(object).where(values.notna())
    return series.astype("string").astype(object).where(series.notna())


def _apply_vocabulary(values: pd.Series, categories) -> pd.Series:
    """Map a column onto its declared domain, folding the rest into OTHER."""
    known = pd.Index(categories)
    text = values.astype("string").astype(object).where(values.notna())
    folded = text.where(text.isin(known) | text.isna(), OTHER)
    return folded.fillna(MISSING).astype(str)


def _days(series: pd.Series) -> np.ndarray:
    """Datetime or period values as float days since :data:`EPOCH`; NaT is NaN."""
    if isinstance(series.dtype, pd.PeriodDtype):
        series = series.dt.to_timestamp()
    when = pd.to_datetime(series)
    if when.dt.tz is not None:
        when = when.dt.tz_convert(None)
    return ((when - EPOCH) / pd.Timedelta(days=1)).to_numpy(float)


def _datetimes(series: pd.Series) -> pd.Series:
    """A datetime or period column as naive timestamps."""
    if isinstance(series.dtype, pd.PeriodDtype):
        series = series.dt.to_timestamp()
    when = pd.to_datetime(series)
    if when.dt.tz is not None:
        when = when.dt.tz_convert(None)
    return when


def _whole_days(series: pd.Series) -> np.ndarray:
    """The date of each timestamp as whole days since :data:`EPOCH`; NaT is NaN."""
    when = _datetimes(series)
    return ((when.dt.normalize() - EPOCH) / pd.Timedelta(days=1)).to_numpy(float)


def _hours(series: pd.Series) -> np.ndarray:
    """The time of day of each timestamp as hours since midnight, in [0, 24); NaT is NaN.

    Taken from the timestamp itself rather than from the fraction of a day
    count, so 13:30 is exactly 13.5 and not what is left of a number near
    twenty thousand.
    """
    when = _datetimes(series)
    return ((when - when.dt.normalize()) / pd.Timedelta(hours=1)).to_numpy(float)


# The two numerical columns a timestamp with a time of day becomes, and how
# each is read off the raw column.
TIMESTAMP_PARTS = (("days", "datetime_days", _whole_days), ("hours", "datetime_hours", _hours))
CONVERTERS = {"datetime": _days, "datetime_days": _whole_days, "datetime_hours": _hours}


def _numbers(series: pd.Series) -> np.ndarray:
    """A numerical column as float64 with ``+-inf`` read as missing."""
    # A copy: pandas may hand out a read-only view of the column's own memory.
    values = np.array(pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64),
                      dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def check_row_subset(name: str, frame: pd.DataFrame, positions: dict, rule: dict) -> None:
    """Raise unless the held rows are whole groups, ``groups_kept`` of them, and nothing else.

    The library-independent half of the subset link: the stored positions
    must be closed under the subset column (a group is held entirely or not
    at all) and hold the recorded number of groups and rows.  Whether they
    are the groups the seeded draw yields is checked by the test suite, not
    here, because the draw itself is what a library could change.

    One table has a subset: ``sepsis_prediction``, one row per ICU hour,
    held as every row of 16,000 of its 31,930 patients (615,404 of 1,228,686
    rows).  ``rule`` is its ``row_subset`` record: ``column`` is
    ``Patient_ID``, ``groups_kept`` 16,000, ``rows_kept`` 615,404.  A split
    that held some of a patient's hours and left out the rest would pass
    :func:`~tabgen_bench.validation.check_partition` and fail here.
    """
    column = rule["column"]
    held = np.concatenate([np.asarray(positions[split], dtype=np.int64) for split in SPLITS])
    labels = _convert_to_string_labels(frame[column]).astype(str).to_numpy()
    kept = set(labels[held])
    mask = np.zeros(len(frame), dtype=bool)
    mask[held] = True
    outside = set(labels[~mask])
    if kept & outside:
        raise SplitIndicesError(
            f"{name}: {len(kept & outside)} groups of {column!r} are partly held and partly "
            "left out; the stored split is not whole groups")
    if len(kept) != rule["groups_kept"] or len(held) != rule["rows_kept"]:
        raise SplitIndicesError(
            f"{name}: the stored split holds {len(kept)} groups of {column!r} in {len(held)} "
            f"rows; tgb_split.json records {rule['groups_kept']} groups in {rule['rows_kept']} rows")


# --------------------------------------------------------------------------
# Preparing a table
# --------------------------------------------------------------------------

def table_description(name: str, metadata: dict, task: dict) -> str:
    text = (f"{name.replace('_', ' ')}: a {metadata['domain_str']} dataset "
            f"({metadata['dataset_source']}, {metadata['dataset_year']}); "
            f"{task['problem_type'].replace('_', ' ')} of "
            f"{task['target_column_name']}")
    return text


def table_metadata(metadata: dict, task: dict) -> dict[str, str]:
    """Schema metadata: the domain and the task."""
    return {
        "domain": metadata["domain_str"],
        "task type": task["problem_type"].replace("_", " "),
        "target": task["target_column_name"],
    }


def apply_dtype_overrides(name: str, frame: pd.DataFrame, dtypes: dict) -> tuple:
    """Recast the columns :data:`registry.DTYPE_OVERRIDES` names for ``name``.

    The first step of preparing a table: ``frame`` and ``dtypes`` are what
    :func:`~tabgen_bench.datasets.read_raw` returned, typed as the bundle
    declares, and come back with each overridden column cast to the dtype
    the benchmark reads it under (:func:`~tabgen_bench.datasets.apply_dtypes`)
    and ``dtypes`` saying so.  An override naming a column the table lacks is
    a ``KeyError``.  A table with no overrides comes back untouched.
    """
    overrides = dtype_overrides(name)
    if not overrides:
        return frame, dtypes
    unknown = sorted(set(overrides) - set(dtypes))
    if unknown:
        raise KeyError(f"{name}: dtype overrides name columns the table lacks: {unknown}")
    recast = apply_dtypes(frame[list(overrides)], overrides)
    frame = frame.copy()
    for column in overrides:
        frame[column] = recast[column]
    return frame, {column: overrides.get(column, dtype) for column, dtype in dtypes.items()}


def _encoders(columns: list) -> dict:
    """How each prepared column is computed, read off a column specification."""
    encoders = {}
    for column in columns:
        if "indicator_of" in column:
            encoders[column["name"]] = ("indicator", column["indicator_of"])
        elif column["kind"] == "categorical":
            encoders[column["name"]] = ("categorical", list(column["categories"]))
        else:
            encoders[column["name"]] = (column["source"], float(column["fill"]))
    return encoders


def raw_column_of(column: dict) -> str:
    """The raw column a prepared column is computed from."""
    return column.get("indicator_of") or column.get("part_of") or column["name"]


def encode_splits(column_spec: dict, splits: dict) -> dict:
    """Encode every split with a column specification; nothing is fitted.

    Returns ``{split: frame}``, each frame in the specification's column
    order with no nulls: categoricals folded onto their vocabularies,
    numbers filled, indicators and timestamp parts computed
    (:func:`_encode_frame`).  This is the whole of a TGB build's encoding;
    the specification it applies is the package's, behind its checksum.
    """
    encoders = _encoders(column_spec["columns"])
    return {split: _encode_frame(frame, column_spec, encoders) for split, frame in splits.items()}


def _encode_frame(frame: pd.DataFrame, spec, encoders) -> pd.DataFrame:
    """Impute, fold and order one split into the schema's column order."""
    out = {}
    for column in spec["columns"]:
        how, payload = encoders[column["name"]]
        if how == "categorical":
            out[column["name"]] = _apply_vocabulary(
                _convert_to_string_labels(frame[column["name"]]), payload).to_numpy()
        elif how == "indicator":
            source = frame[payload]
            values = (_days(source) if kind_of(str(source.dtype)) == "datetime"
                      else _numbers(source))
            out[column["name"]] = np.where(np.isfinite(values), "present", "missing")
        elif how in CONVERTERS:
            out[column["name"]] = np.nan_to_num(
                CONVERTERS[how](frame[raw_column_of(column)]), nan=payload)
        else:
            out[column["name"]] = np.nan_to_num(_numbers(frame[column["name"]]), nan=payload)
    return pd.DataFrame(out, columns=spec["column_order"])


def load_raw_splits(name: str, data_root: Path = DATA_ROOT, *, verify: bool = True) -> dict:
    """Links 1 to 4: the raw table, typed, and its stored split, both checked.

    Returns a dict: ``frame`` (the raw table under the benchmark's dtypes),
    ``dtypes``, ``task`` and ``metadata`` from the bundle, ``splits``
    (``{split: frame}`` taken by the stored positions), ``how`` (which stored
    split, with its sha256 and rule, and the row subset when the table has
    one) and ``verified`` (the raw verification report, or None with
    ``verify=False``).  Both a build and the creator's recording start here.
    """
    # Links 1 and 2: the raw schema, then the raw-data checksum.
    verified = verify_source(name, data_root) if verify else None
    raw = Path(data_root) / "raw"
    frame, dtypes, task, metadata = read_raw(name, raw)
    frame, dtypes = apply_dtype_overrides(name, frame, dtypes)

    # Links 3 and 4: the stored row indices, behind the split-file checksum.
    # Nothing is drawn: the split is the package's, and it has no row cap.
    record, positions = load_split_indices(name)
    if record["raw_rows"] != len(frame):
        raise SplitIndicesError(
            f"{name}: the stored split is for {record['raw_rows']} raw rows, the "
            f"table read has {len(frame)}")
    how = {"split": "tgb", "indices_file": TGB_SPLIT_INDICES,
           "indices_sha256": record["sha256"], "rule": dict(record.get("rule") or {})}
    # A table the benchmark holds a whole-group subset of: the stored split
    # must be that subset, and the registry's rule must still say so.
    subset_rule = row_subset(name)
    stored_subset = record.get("row_subset")
    if bool(stored_subset) != bool(subset_rule) or (
            stored_subset and any(stored_subset[k] != subset_rule[k]
                                  for k in ("column", "groups", "seed"))):
        raise SplitIndicesError(
            f"{name}: the stored split records the row subset {stored_subset}, "
            f"the registry's ROW_SUBSETS says {subset_rule}")
    if stored_subset:
        check_row_subset(name, frame, positions, stored_subset)
        how["row_subset"] = stored_subset
    splits = {split: frame.iloc[idx].reset_index(drop=True)
              for split, idx in positions.items()}
    return {"frame": frame, "dtypes": dtypes, "task": task, "metadata": metadata,
            "splits": splits, "how": how, "verified": verified}


def prepare_table(name: str, data_root: Path = DATA_ROOT, *,
                  verify: bool = True, strict: bool = True) -> dict:
    """Build one table of the benchmark and write ``prepared/<name>/``.

    The build follows the reproducibility chain link by link and fits
    nothing:

    1. the raw table must have the raw schema the benchmark records, and
    2. every raw file the recorded sha256 (:func:`verify_source`; skipped
       with ``verify=False``, which the spec then says);
    3. the split is the row positions stored in the package, and
    4. that file must have its recorded sha256 and partition the raw rows
       (:func:`load_split_indices`); for a table the benchmark holds a row
       subset of, the held rows must be whole groups, the recorded number of
       them (:func:`check_row_subset`);
    5. the splits are taken by position and encoded with the column
       specification stored in the package -- vocabularies, fills, indicators
       and dropped columns -- which is taken behind its own sha256 and applied
       as it is (:func:`load_column_spec`, :func:`encode_splits`);
    6. each prepared split's content digest must equal the one the benchmark
       ships (:func:`verify_prepared_dataframes`), read back from the files
       just written.  With ``strict`` a difference removes them and raises;
       without it the table stays and its spec says ``matches_tgb: false``.

    The spec records what was checked under ``reproducibility``.  A null is
    never a reason to drop a row: a numerical null is median-filled, with a
    ``<col>__missing`` column beside it once the rate reaches
    :data:`MISSING_INDICATOR_MIN_FRAC`, and a categorical null is the level
    ``(missing)``.  Fitting a column specification, for a new table or
    after a policy change, is the creator's job:
    :func:`tabgen_bench.creator_tool.record_columns`.
    """
    loaded = load_raw_splits(name, data_root, verify=verify)
    frame, dtypes, task, metadata = (loaded[k] for k in ("frame", "dtypes", "task", "metadata"))
    splits, how, verified = loaded["splits"], loaded["how"], loaded["verified"]
    raw = Path(data_root) / "raw"

    # Link 5: the stored column specification, applied as it is.
    column_spec = load_column_spec(name)
    spec = {"columns": [dict(column) for column in column_spec["columns"]],
            "column_order": list(column_spec["column_order"]),
            "dropped": dict(column_spec["dropped"])}
    encoded = encode_splits(spec, splits)
    digests = {s: compute_dataframe_digest(f) for s, f in encoded.items()}
    directory = table_dir(data_root, name)
    directory.mkdir(parents=True, exist_ok=True)
    for split, f in encoded.items():
        f.to_parquet(split_file(data_root, name, split), index=False)
    # Link 6: the content of what was prepared, read back from the files just
    # written, against the digests the benchmark ships.
    try:
        verify_prepared_dataframes(name, data_root)
        matches_tgb = True
    except PreparedMismatchError:
        if strict:
            for split in encoded:            # leave nothing that is not the benchmark's
                split_file(data_root, name, split).unlink(missing_ok=True)
            raise
        matches_tgb = False
    spec.update({
        "table": name,
        "unique_name": metadata["unique_name"],
        "description": table_description(name, metadata, task),
        "metadata": table_metadata(metadata, task),
        "task_type": task["problem_type"],
        "target": task["target_column_name"],
        "objective_metric": task.get("objective_metric_name"),
        "domain": metadata["domain_str"],
        "source": metadata["dataset_source"],
        "year": metadata["dataset_year"],
        "license": metadata.get("license"),
        "reference": metadata.get("academic_reference_bibtex_key"),
        "container_uuid": json.loads(
            (dataset_dir(name, raw) / DOWNLOAD_FILES["container"]).read_text())["uuid"],
        "split": "tgb",
        "tgb": True,
        "digest_label": DIGEST_LABEL,
        "digests": digests,
        "reproducibility": {
            "raw_schema_verified": verified is not None,
            "raw_checksum_verified": verified is not None,
            "split_checksum_verified": True,
            "column_spec_verified": True,
            "matches_tgb": matches_tgb,
        },
        "rows": {s: int(len(f)) for s, f in encoded.items()},
        "raw_rows": int(len(frame)),
        "raw_columns": int(len(dtypes)),
        "max_categories": MAX_CATEGORIES,
        "dtype_overrides": dtype_overrides(name),
        "row_subset": how.get("row_subset"),
        "missing_indicator_min_frac": MISSING_INDICATOR_MIN_FRAC,
    })
    # How the split was made.  For the benchmark it names the package's
    # ``tgb_split_indices.parquet`` and its sha256: the split is that file,
    # and nothing of it is written again here.
    # How the split was made: the package's ``tgb_split_indices.parquet`` and
    # its sha256.  The split is that file, and nothing of it is written here.
    spec["split_record"] = how
    spec_file(data_root, name).write_text(json.dumps(spec, indent=1))
    return spec


# --------------------------------------------------------------------------
# Preparing the corpus
# --------------------------------------------------------------------------

def prepare(data_root: Path = DATA_ROOT, names=None, *, force: bool = False,
            verify_sources: bool = True, **options) -> dict:
    """Download and prepare ``names``; write the manifest.

    ``names`` defaults to the benchmark's tables (:func:`benchmark_tables`).
    A name that is not one of them is not downloaded (:func:`download` skips
    it with a message), so it is built only if its raw files are already on
    disk.

    Tables already prepared are skipped unless ``force``; a prepared table's
    own spec records the options it was built with, so a corpus prepared in two
    sittings with different options is visible in ``python -m tabgen_bench.datasets``.  With
    ``verify_sources`` every table that has an origin record is checked
    against it (:func:`verify_source`) before it is built.  ``options`` go to
    :func:`prepare_table`.
    """
    data_root = Path(data_root)
    names = list(names) if names else list(benchmark_tables())
    todo = [n for n in names if force or not spec_file(data_root, n).exists()]
    # Only what raw/ lacks is fetched, so a second data root whose raw/ is a
    # link to the first (an uncapped re-cut, say) never touches the network.
    missing = [n for n in todo if not raw_present(n, data_root)]
    if missing:
        download(missing, data_root)
    for i, name in enumerate(todo, 1):
        spec = prepare_table(name, data_root, verify=verify_sources, **options)
        print(f"[{i}/{len(todo)}] {name}: {spec['rows']} rows, "
              f"{len(spec['column_order'])} columns, "
              f"dropped {len(spec['dropped'])}", flush=True)
    return write_manifest(data_root)


def prepare_tables(tables=None, *, data_root: Path | None = None,
                   force: bool = False) -> dict:
    """Build the benchmark's prepared tables, as TGB.

    ``tables`` is a list of table names; None means every table of the
    benchmark.  Each table is downloaded if it is not on disk, then taken
    through the whole reproducibility chain: TGB raw schema, raw-data
    checksum, the stored split behind its own checksum, and the content
    digest of each prepared split.  A table that fails any link raises a
    :class:`~tabgen_bench.validation.ReproducibilityError` and is not
    written.  Tables already prepared are skipped unless ``force``.

    Writes ``<data_root>/prepared/<table>/`` and returns the manifest, a
    dict with one summary entry per prepared table.  This function has no
    split or column options on purpose: it builds the benchmark and nothing
    else.  :func:`prepare` and :func:`prepare_table` take them.
    """
    return prepare(data_root or default_data_root(), tables, force=force)


def _summary(spec: dict) -> dict:
    """The manifest's per-table entry: enough to select tables without the spec."""
    categorical = [c for c in spec["columns"] if c["kind"] == "categorical"]
    return {
        "task_type": spec["task_type"],
        "target": spec["target"],
        "split": spec.get("split", "tgb"),
        "tgb": spec.get("tgb", False),
        "matches_tgb": spec.get("reproducibility", {}).get("matches_tgb"),
        "columns": len(spec["column_order"]),
        "numerical": len(spec["columns"]) - len(categorical),
        "categorical": len(categorical),
        "max_cardinality": max((len(c["categories"]) for c in categorical), default=0),
        "rows": spec["rows"],
        "raw_rows": spec["raw_rows"],
        "raw_columns": spec["raw_columns"],
        "dropped": len(spec["dropped"]),
        "max_categories": spec["max_categories"],
    }


def write_manifest(data_root: Path = DATA_ROOT) -> dict:
    """Index every ``prepared/<name>/spec.json`` into ``prepared/manifest.json``."""
    prepared = Path(data_root) / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    tables = {}
    for spec_path in sorted(prepared.glob("*/spec.json")):
        spec = json.loads(spec_path.read_text())
        tables[spec["table"]] = _summary(spec)
    manifest = {
        "source": "BeyondArena (Purucker et al., 2026, arXiv:2606.30410), "
                  f"as published at {HF_REPO}",
        "url": HF_URL,
        "epoch": str(EPOCH.date()),
        "tables": tables,
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="tables to prepare (default: the benchmark's "
                             "tables, registry.TABLES)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--no-strict", action="store_true",
                        help="tgb only: write a table whose content digest "
                             "differs from the benchmark's instead of stopping")
    parser.add_argument("--force", action="store_true",
                        help="re-prepare tables that are already prepared")
    parser.add_argument("--no-verify-sources", action="store_true",
                        help="skip the check of raw files against "
                             "table_sources/<table>/files.json (and so write "
                             "no verified.json)")
    args = parser.parse_args()
    prepare(args.data_root, args.datasets, force=args.force,
            verify_sources=not args.no_verify_sources,
            strict=not args.no_strict)


if __name__ == "__main__":
    main()
