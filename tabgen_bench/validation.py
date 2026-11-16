"""Validation: every check of the reproducibility chain, and its evidence.

A TGB build of a table passes four checks before any row is used, one
before any column is encoded and one after
(:func:`tabgen_bench.preprocessing.prepare_table` runs them in this order)::

    TGB raw schema      verify_raw_schema         schema.json
            |
    raw-data checksum         verify_source         files.json
            |
    stored row indices        load_split_indices    tgb_split_indices.parquet
            |
    split-file checksum       load_split_indices    tgb_split.json
            |
    train / validation / test
            |
    stored column spec        load_column_spec      tgb_columns.json
            |
    content digests           verify_prepared_dataframes   tgb_prepared.json

* :func:`verify_raw_schema` reads only the raw table's metadata: its columns, in
  order, with the dtypes BeyondArena declares, and its row count must be the
  ones the benchmark records.  It is cheap, and it fails with the name of the
  column that differs rather than with an opaque hash.
* :func:`verify_source` hashes every raw file: each must have the recorded
  size and sha256.  It runs the schema check first.  When both pass it writes
  ``<data root>/table_sources/<table>/verified.json`` -- evidence about one
  machine's copy, so it is never shipped -- and when either fails it removes
  that file and raises.
* :func:`load_split_indices` is the only checked way to the stored split.  It
  verifies the shipped file's sha256, that the split refers to the raw table
  the benchmark pins, and that the positions partition the raw rows: every
  row in exactly one split.
* :func:`load_column_spec` is the checked way to the stored column
  specification: the vocabularies, fills and dropped columns a TGB
  build applies instead of fitting.  Fitting is where library versions can
  disagree -- which labels survive a tie at the vocabulary limit depends on
  how a library orders equal counts -- so the benchmark stores the outcome.
* :func:`compute_dataframe_digest` defines a content digest of a prepared split that does
  not depend on the parquet writer, and :func:`verify_prepared_dataframes` compares the
  digests of what a machine prepared with the ones the benchmark ships.

Every failure is a :class:`ReproducibilityError` naming the table and what
differed.

    python -m tabgen_bench.validation                     # raw data of every table
    python -m tabgen_bench.validation --prepared          # ... and the prepared splits
    python -m tabgen_bench.validation --datasets sdss_17 kick
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import (
    DATA_ROOT,
    DOWNLOAD_FILES,
    SPLITS,
    benchmark_tables,
    default_data_root,
    file_sha256,
    prepared_dir,
    raw_revision,
    TGB_SECTIONS,
    read_source,
    section_text,
    read_split_indices_file,
    source_dir,
    split_indices_path,
)

# What the check found on *this* machine.  Never shipped: it is written under
# the data root, ``<data root>/table_sources/<table>/verified.json``, once a
# user's own download of the table has been compared with the shipped record.
VERIFIED = "verified.json"

# Names the recipe :func:`compute_dataframe_digest` follows, and goes into every digest it
# makes; recorded beside the digests.  Bumped whenever the recipe would give a
# different value for the same frame.
DIGEST_LABEL = "tabgen-bench frame digest v1"


class ReproducibilityError(ValueError):
    """A link of the reproducibility chain did not hold."""


class SchemaMismatchError(ReproducibilityError):
    """The raw table does not have the TGB raw schema."""


class ChecksumMismatchError(ReproducibilityError):
    """A file does not have the recorded size or sha256."""


class SplitIndicesError(ReproducibilityError):
    """The stored split is not usable: wrong file, wrong table, or not a partition."""


class ColumnSpecError(ReproducibilityError):
    """The stored column specification is not the one the benchmark records."""


class PreparedMismatchError(ReproducibilityError):
    """A prepared split does not have the TGB content digest."""


def verified_path(name: str, data_root: Path) -> Path:
    return Path(data_root) / "table_sources" / name / VERIFIED


# A file's modification time is trusted only when it is at least this much
# older than the verification that recorded it.  Timestamps are coarse (a
# kernel tick, or a second on some filesystems), so a file written in the same
# instant it was verified could be changed again without its time moving.
MTIME_MARGIN_NS = 2_000_000_000


def record_digest(source: dict) -> str:
    """sha256 of what a raw verification is checked against, in ``source``.

    The pinned commit, the file list with its sizes and checksums, and the
    TGB raw schema.  A stored verification made against another record
    says nothing about this one.
    """
    text = section_text({"revision": source["origin"]["revision"],
                         "files": source["files"], "schema": source["schema"]})
    return hashlib.sha256(text.encode()).hexdigest()


def reusable_verification(name: str, data_root: Path, source: dict) -> dict | None:
    """The stored ``verified.json`` of ``name`` if it still holds, else None.

    It holds when it was made against the same record (:func:`record_digest`)
    and the same ``raw`` directory, and every data file is as it was when
    verified: the same size and the same modification time, that time being
    earlier than the verification.  A file modified since has a later time,
    and a file replaced by a copy has another one; either way None is
    returned and the caller verifies again.
    """
    target = verified_path(name, data_root)
    try:
        stored = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    raw = Path(data_root) / "raw"
    started = stored.get("verified_at_ns")
    if (not isinstance(started, int) or stored.get("record_sha256") != record_digest(source)
            or stored.get("raw") != str(raw.resolve())
            or not stored.get("all_files_match_record")):
        return None
    seen = {entry.get("path"): entry for entry in stored.get("files", [])}
    for entry in source["files"]:
        was = seen.get(entry["path"])
        try:
            now = (raw / entry["path"]).stat()
        except OSError:
            return None
        if (was is None or was.get("mtime_ns") != now.st_mtime_ns
                or was.get("size") != now.st_size
                or now.st_mtime_ns > started - MTIME_MARGIN_NS):
            return None
    return stored


def verification_line(name: str, report: dict | None, data_root: Path) -> str:
    """One status line for ``report``, as the command-line tools print it."""
    if report is None:
        return f"{name}: no origin record, not verified"
    where = f"{report['repo_id']}@{report['revision'][:12]} -> {verified_path(name, data_root)}"
    if report.get("reused"):
        return (f"{name}: {report['files_checked']} files unchanged since they were "
                f"verified at {report['verified_at']}, {where}")
    return f"{name}: {report['files_checked']} files match {where}"


# --------------------------------------------------------------------------
# 1. Raw schema validation
# --------------------------------------------------------------------------

def _raw_file(source: dict, raw: Path, key: str) -> Path:
    """The raw file ``key`` (a key of ``DOWNLOAD_FILES``) of a recorded table."""
    for entry in source["files"]:
        if entry["name"] == DOWNLOAD_FILES[key]:
            return Path(raw) / entry["path"]
    raise KeyError(f"{source['origin']['table']}: no {DOWNLOAD_FILES[key]} in its files record")


def verify_raw_schema(name: str, data_root: Path = DATA_ROOT) -> dict | None:
    """Check the raw table of ``name`` against its raw schema.

    The raw ``dataset.parquet`` must hold the recorded number of rows and the
    recorded columns in the recorded order, and the bundle's ``dtypes.json``
    must declare the recorded dtype for each.  Only metadata is read.  Returns
    ``{"rows", "columns", "matches_record": True}``, or None for a table
    without a record; raises :class:`SchemaMismatchError` naming the first
    difference.
    """
    source = read_source(name)
    if source is None:
        return None
    import pyarrow.parquet as pq

    raw = Path(data_root) / "raw"
    schema = source["schema"]
    table_file, dtypes_file = _raw_file(source, raw, "table"), _raw_file(source, raw, "dtypes")
    # Both files the schema is read from must be on disk.
    for path in (table_file, dtypes_file):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path} is missing; run "
                                    f"python -m tabgen_bench.download --datasets {name}")
    parquet = pq.ParquetFile(table_file)
    found, expected = list(parquet.schema_arrow.names), [c["name"] for c in schema["columns"]]
    # The row count, from the parquet footer.
    if parquet.metadata.num_rows != schema["rows"]:
        raise SchemaMismatchError(
            f"{name}: the raw table has {parquet.metadata.num_rows} rows, the raw "
            f"schema {schema['rows']}")
    # The column names, in file order, from the parquet's own schema.
    if found != expected:
        absent = [c for c in expected if c not in found]
        extra = [c for c in found if c not in expected]
        detail = (f"missing {absent[:5]}, unexpected {extra[:5]}" if absent or extra else
                  next(f"position {i} holds {a!r}, the schema says {b!r}"
                       for i, (a, b) in enumerate(zip(found, expected)) if a != b))
        raise SchemaMismatchError(f"{name}: the raw columns differ from the raw "
                                  f"schema: {detail}")
    # The dtype BeyondArena declares for each column, from dtypes.json.
    declared = json.loads(dtypes_file.read_text())
    for column in schema["columns"]:
        if declared.get(column["name"]) != column["dtype"]:
            raise SchemaMismatchError(
                f"{name}: column {column['name']!r} is declared "
                f"{declared.get(column['name'])!r}, the raw schema says "
                f"{column['dtype']!r}")
    return {"rows": int(schema["rows"]), "columns": len(expected), "matches_record": True}


# --------------------------------------------------------------------------
# 2. Raw-data checksum
# --------------------------------------------------------------------------

def verify_source(name: str, data_root: Path = DATA_ROOT, *,
                  recheck: bool = False) -> dict | None:
    """Check the downloaded ``raw/<name>`` against its record.

    The raw table must have the  raw schema (:func:`verify_raw_schema`),
    and every recorded file must be present at the recorded path with the
    recorded size and sha256.  When they all are, the outcome is written to
    ``<data_root>/table_sources/<name>/verified.json`` -- when it was checked,
    the commit it was checked against, the schema it matched, each file's
    measured size, sha256 and modification time, and the commit the download
    cache says the files came from -- and returned.  On a mismatch any earlier
    ``verified.json`` is removed and a :class:`ReproducibilityError` names
    what differed: the table on disk is not the one the benchmark was
    defined on.  A table without a record returns None and writes nothing.

    A table already verified is not hashed again while its files are
    untouched: when ``verified.json`` was made against the same record and
    every file still has the size and modification time recorded in it, that
    time being earlier than the verification, the stored report is returned
    as it is (:func:`reusable_verification`).  A file modified after the
    verification, or replaced, makes the whole table be verified again.
    This is a shortcut on the files' timestamps, not a proof: a file altered
    with its modification time put back is only caught by ``recheck``.

    Args:
        name: The table.
        data_root: The data root holding ``raw/``.
        recheck: Verify from the files whatever ``verified.json`` says.

    Returns:
        The report written to ``verified.json``, with one more key that is
        not stored: ``reused``, True when the stored report was returned
        without reading the files and False when they were just verified.
        None for a table without a record.
    """
    source = read_source(name)
    if source is None:
        return None
    raw = Path(data_root) / "raw"
    target = verified_path(name, data_root)
    origin = source["origin"]
    # The shortcut: an earlier report still holds while the files are untouched.
    if not recheck:
        stored = reusable_verification(name, data_root, source)
        if stored is not None:
            return {**stored, "reused": True}
    started = time.time_ns()
    files, problem, schema = [], None, None
    # Every recorded file must be present, before anything is read.
    for entry in source["files"]:
        if not (raw / entry["path"]).exists():
            problem = FileNotFoundError(
                f"{name}: {entry['path']} is missing from {raw}; the benchmark pins "
                f"{origin['repo_id']}@{origin['revision'][:12]}")
            break
    # The raw schema check: cheap, and it names the column that differs.
    if problem is None:
        try:
            schema = verify_raw_schema(name, data_root)
        except ReproducibilityError as error:
            problem = error
    # The checksums check: each file's size, then its sha256, against files.json.
    if problem is None:
        for entry in source["files"]:
            path = raw / entry["path"]
            # Read before hashing: a write during the hash moves the time on,
            # so the next call sees a file that differs from this record.
            stat = path.stat()
            size = stat.st_size
            if size != entry["size"]:
                problem = ChecksumMismatchError(
                    f"{name}: {entry['path']} is {size} bytes, the origin record says "
                    f"{entry['size']}")
                break
            digest = file_sha256(path)
            if digest != entry["sha256"]:
                problem = ChecksumMismatchError(
                    f"{name}: {entry['path']} does not match the sha256 of its origin "
                    f"record ({source_dir(name) / 'files.json'})")
                break
            files.append({"path": entry["path"], "size": size, "sha256": digest,
                          "mtime_ns": stat.st_mtime_ns, "matches_record": True})
    # Any failure removes the old evidence and raises; nothing is reported.
    if problem is not None:
        target.unlink(missing_ok=True)
        raise problem
    # The evidence: what was checked, when, and each file's size, sha256 and
    # modification time, written under the data root for the next call to reuse.
    cached = raw_revision(name, raw)
    report = {
        "table": name,
        # When the verification began, for a reader and in nanoseconds since
        # the epoch; files are compared against the second one.
        "verified_at": pd.Timestamp(started, unit="ns", tz="UTC").isoformat(timespec="seconds"),
        "verified_at_ns": started,
        "repo_id": origin["repo_id"],
        "revision": origin["revision"],
        "record_sha256": record_digest(source),
        "raw": str(raw.resolve()),
        "method": "raw schema, then size and sha256 of every file against "
                  "table_sources/<table>/files.json",
        "schema": schema,
        "files_checked": len(files),
        "all_files_match_record": True,
        "download_cache_revision": cached,
        "download_cache_agrees": (cached == origin["revision"]) if cached else None,
        "files": files,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1) + "\n")
    return {**report, "reused": False}


# --------------------------------------------------------------------------
# 3-4. Stored row indices and the split-file checksum
# --------------------------------------------------------------------------

def check_partition(name: str, positions: dict, raw_rows: int, *,
                    held_rows: int | None = None) -> None:
    """Raise unless ``positions`` puts each of ``raw_rows`` rows in exactly one split.

    For a table the benchmark holds a row subset of, ``held_rows`` is the
    size of that subset: the positions must then be ``held_rows`` distinct
    rows of the table, each in exactly one split, and the rest in none.
    """
    # Every position of every split, as one array.
    held = np.concatenate([np.asarray(positions[split], dtype=np.int64) for split in SPLITS])
    # The count: every raw row, or every row of the subset.
    expected = raw_rows if held_rows is None else held_rows
    if len(held) != expected:
        raise SplitIndicesError(f"{name}: the split holds {len(held)} positions, "
                                f"{expected} expected for a raw table of {raw_rows} rows")
    # The range: a position is a row number of the raw table.
    if len(held) and (held.min() < 0 or held.max() >= raw_rows):
        raise SplitIndicesError(f"{name}: the split holds positions outside 0..{raw_rows - 1}")
    # No row in two splits; and, unless a subset is held, no row in none.
    counts = np.bincount(held, minlength=raw_rows)
    if (counts > 1).any():
        raise SplitIndicesError(
            f"{name}: {int((counts > 1).sum())} raw rows are in more than one split")
    if held_rows is None and (counts == 0).any():
        raise SplitIndicesError(f"{name}: {int((counts == 0).sum())} raw rows are in no split")


def load_split_indices(name: str) -> tuple[dict, dict]:
    """The TGB split of ``name``: ``(record, {split: raw row positions})``.

    The checked way to the stored split.  Before the positions are returned:

    * the shipped ``tgb_split_indices.parquet`` must have the sha256 that
      ``tgb_split.json`` records (the split-file checksum);
    * ``tgb_split.json`` must refer to the raw table the benchmark pins -- the
      same sha256 and row count as ``files.json`` and ``schema.json``;
    * the positions must partition the raw rows, and each split must hold the
      recorded number of rows.

    Positions are ascending within each split, which is the row order of the
    prepared files.  Raises :class:`SplitIndicesError` on any failure, and
    ``KeyError`` for a table without a TGB split.
    """
    # The record: tgb_split.json, which must exist for the table.
    source = read_source(name)
    if source is None or "split" not in source:
        raise KeyError(f"{name!r} has no TGB split in the benchmark's records")
    record = source["split"]
    # The file: tgb_split_indices.parquet, which must be in the package.
    path = split_indices_path(name)
    if not path.exists():
        raise SplitIndicesError(f"{name}: {path} is missing from the package")
    # The split-file checksum: the file is the one the record vouches for.
    digest = file_sha256(path)
    if digest != record["sha256"]:
        raise SplitIndicesError(
            f"{name}: {path.name} has sha256 {digest[:16]}..., tgb_split.json records "
            f"{record['sha256'][:16]}...; the stored split is not the benchmark's")
    # The record must refer to the raw table the benchmark pins: same sha256 ...
    raw_sha = next(f["sha256"] for f in source["files"] if f["name"] == DOWNLOAD_FILES["table"])
    if record["raw_sha256"] != raw_sha:
        raise SplitIndicesError(
            f"{name}: the stored split was made for a raw table with sha256 "
            f"{record['raw_sha256'][:16]}..., the benchmark pins {raw_sha[:16]}...")
    # ... and the same row count as the raw schema.
    if record["raw_rows"] != source["schema"]["rows"]:
        raise SplitIndicesError(
            f"{name}: the stored split is for {record['raw_rows']} raw rows, the raw "
            f"schema has {source['schema']['rows']}")
    # The stored row indices: read, and checked to partition the raw rows
    # (or the rows of the table's row subset).
    positions = read_split_indices_file(path)
    subset = record.get("row_subset")
    check_partition(name, positions, record["raw_rows"],
                    held_rows=subset["rows_kept"] if subset else None)
    # Each split must hold the number of rows the record says.
    rows = {split: int(len(positions[split])) for split in SPLITS}
    if rows != record["rows"]:
        raise SplitIndicesError(f"{name}: the stored split holds {rows}, tgb_split.json "
                                f"records {record['rows']}")
    return record, positions


# --------------------------------------------------------------------------
# 5. Stored column specification
# --------------------------------------------------------------------------

def load_column_spec(name: str) -> dict:
    """The TGB column specification of ``name``, behind its checksum.

    Returns ``{"column_order", "columns", "dropped"}``: the prepared columns
    in order, each with its kind and either its vocabulary, its fill value or
    the column it is the missing-indicator of, the raw timestamp a ``__days``
    or ``__hours`` column is part of, and the raw columns that are dropped
    and why.  The shipped ``tgb_columns.json`` must have the sha256 that
    ``tgb_prepared.json`` records, its columns must be the ones it orders, and
    every raw column it reads must be in the TGB raw schema.  Raises
    :class:`ColumnSpecError` otherwise, and ``KeyError`` for a table without
    one.
    """
    # The records: tgb_columns.json and, for its checksum, tgb_prepared.json.
    source = read_source(name)
    if source is None or "columns" not in source or "prepared" not in source:
        raise KeyError(f"{name!r} has no TGB column specification in the "
                       f"benchmark's records")
    # The checksum: tgb_columns.json must hash to what tgb_prepared.json records.
    path = source_dir(name) / TGB_SECTIONS["columns"]
    digest = file_sha256(path)
    if digest != source["prepared"]["columns_sha256"]:
        raise ColumnSpecError(
            f"{name}: tgb_columns.json has sha256 {digest[:16]}..., tgb_prepared.json records "
            f"{source['prepared']['columns_sha256'][:16]}...; the stored column "
            f"specification is not the benchmark's")
    # Internal consistency: the columns defined are the columns ordered, once each.
    spec = source["columns"]
    names = [column["name"] for column in spec["columns"]]
    if names != spec["column_order"] or len(set(names)) != len(names):
        raise ColumnSpecError(f"{name}: tgb_columns.json orders columns it does not define")
    # Against the raw schema: every prepared column reads a raw column that exists,
    # itself, or the column it is the missing-indicator of or a timestamp part of.
    raw_columns = {column["name"] for column in source["schema"]["columns"]}
    for column in spec["columns"]:
        reads = column.get("indicator_of") or column.get("part_of") or column["name"]
        if reads not in raw_columns:
            raise ColumnSpecError(f"{name}: prepared column {column['name']!r} reads "
                                  f"{reads!r}, which the raw schema lacks")
    # The three sections a build applies; the table's name is left out.
    return {key: spec[key] for key in ("column_order", "columns", "dropped")}


# --------------------------------------------------------------------------
# 6. Content digests of the prepared splits
# --------------------------------------------------------------------------

def compute_dataframe_digest(frame: pd.DataFrame) -> str:
    """A sha256 of a prepared split's content, independent of how it is stored.

    Two frames get the same digest exactly when they have the same columns in
    the same order, the same number of rows, and the same value in every
    cell.  The digest does not depend on the parquet writer, on compression,
    or on whether pandas holds strings as objects or as arrow strings, so it
    can be compared across machines and library versions.

    It is the sha256 of, in order: the method name; the row and column
    counts; and for each column its name, a kind byte, and its values.  A
    numerical column contributes its values as little-endian float64, with
    every NaN written as the one TGB NaN.  Any other column is read as
    text and contributes the cumulative byte lengths of its values as
    little-endian int64 followed by the UTF-8 bytes of all values joined.  A
    null in a text column is an error: prepared splits have none.
    """
    import pyarrow as pa

    digest = hashlib.sha256()
    # The recipe's label, then the shape: a digest from another recipe or of
    # another shape can never collide with this one.
    digest.update(DIGEST_LABEL.encode() + b"\0")
    digest.update(np.asarray([len(frame), frame.shape[1]], dtype="<i8").tobytes())
    # Then each column in order: its name, a kind byte, and its values.
    for column in frame.columns:
        values = frame[column]
        digest.update(str(column).encode("utf-8") + b"\0")
        # Text (labels, and booleans): every value as UTF-8, through arrow.
        if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(values):
            array = pa.array(values, type=pa.large_string(), from_pandas=True)
            if array.null_count:
                raise ValueError(f"column {column!r} holds {array.null_count} nulls; a "
                                 f"prepared split has none")
            # pandas may hold the column in several chunks, each a slice of a
            # larger buffer.  Read every chunk through its own offsets so the
            # digest sees one run of lengths and one run of bytes either way.
            chunks = array.chunks if isinstance(array, pa.ChunkedArray) else [array]
            ends, texts, total = [np.zeros(1, dtype="<i8")], [], 0
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                _, offsets, data = chunk.buffers()
                offsets = np.frombuffer(offsets, dtype="<i8")[
                    chunk.offset:chunk.offset + len(chunk) + 1]
                start, stop = int(offsets[0]), int(offsets[-1])
                ends.append(offsets[1:] - start + total)
                if stop > start:
                    texts.append(memoryview(data)[start:stop])
                total += stop - start
            # The cumulative lengths first, then the bytes: "ab"+"c" and
            # "a"+"bc" must not collide.
            digest.update(b"s")
            digest.update(np.concatenate(ends).astype("<i8", copy=False).tobytes())
            for text in texts:
                digest.update(text)
        # Numbers: every value as little-endian float64, with one bit pattern
        # for every NaN.
        else:
            numbers = np.array(values.to_numpy(dtype="float64", na_value=np.nan),
                               dtype="<f8", copy=True)
            numbers[np.isnan(numbers)] = np.nan
            digest.update(b"f")
            digest.update(numbers.tobytes())
    return digest.hexdigest()


def prepared_dataframe_digests(name: str, data_root: Path = DATA_ROOT) -> dict:
    """``{split: compute_dataframe_digest}`` of the prepared splits of ``name`` on disk."""
    directory = prepared_dir(data_root, name)
    digests = {}
    for split in SPLITS:
        path = directory / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path} is missing; run "
                                    f"python -m tabgen_bench.preprocessing --datasets {name}")
        digests[split] = compute_dataframe_digest(pd.read_parquet(path))
    return digests


PREPARED_VERIFIED = "prepared_verified.json"


def prepared_verified_path(name: str, data_root: Path) -> Path:
    return Path(data_root) / "table_sources" / name / PREPARED_VERIFIED


def _prepared_files(name: str, data_root: Path) -> dict:
    return {split: prepared_dir(data_root, name) / f"{split}.parquet" for split in SPLITS}


def reusable_prepared_verification(name: str, data_root: Path, expected: dict) -> dict | None:
    """The stored ``prepared_verified.json`` of ``name`` if it still holds, else None.

    The prepared counterpart of :func:`reusable_verification`: the report
    must have been made against the same ``tgb_prepared.json`` record, and
    each prepared parquet must have the size and modification time it had
    when its digest was checked, that time being earlier than the check.
    """
    target = prepared_verified_path(name, data_root)
    try:
        stored = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    started = stored.get("verified_at_ns")
    record = hashlib.sha256(section_text(expected).encode()).hexdigest()
    if (not isinstance(started, int) or stored.get("record_sha256") != record
            or not stored.get("matches_tgb")):
        return None
    seen = {entry.get("path"): entry for entry in stored.get("files", [])}
    for split, path in _prepared_files(name, data_root).items():
        was = seen.get(str(path))
        try:
            now = path.stat()
        except OSError:
            return None
        if (was is None or was.get("mtime_ns") != now.st_mtime_ns
                or was.get("size") != now.st_size
                or now.st_mtime_ns > started - MTIME_MARGIN_NS):
            return None
    return stored


def verify_prepared_dataframes(name: str, data_root: Path = DATA_ROOT, *,
                               recheck: bool = False) -> dict:
    """Check the prepared splits of ``name`` on disk against the TGB digests.

    The three prepared parquet files under ``data_root`` are read and
    digested (:func:`prepared_dataframe_digests`) and each digest must be the
    one the benchmark ships.  A table whose files are untouched since they
    were last checked is not digested again: the stored report under
    ``<data_root>/table_sources/<name>/prepared_verified.json`` is returned
    as it is (:func:`reusable_prepared_verification`), unless ``recheck``.
    A file modified or replaced since makes the table be checked again.

    Returns ``{"digests", "matches_tgb": True, "reused"}``.  Raises
    :class:`PreparedMismatchError` naming the splits that differ,
    ``FileNotFoundError`` for a split that is not on disk, and ``KeyError``
    for a table without TGB digests.  A failure removes the stored report.
    """
    # The record: tgb_prepared.json, made by the same recipe as these digests.
    source = read_source(name)
    if source is None or "prepared" not in source:
        raise KeyError(f"{name!r} has no TGB digests in the benchmark's records")
    expected = source["prepared"]
    target = prepared_verified_path(name, data_root)
    # The shortcut: an earlier check still holds while the files are untouched.
    if not recheck:
        stored = reusable_prepared_verification(name, data_root, expected)
        if stored is not None:
            return {"digests": dict(stored["digests"]), "matches_tgb": True, "reused": True}
    # Read the files' times before digesting them (see verify_source).
    started = time.time_ns()
    files = []
    for split, path in _prepared_files(name, data_root).items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path} is missing; run "
                                    f"python -m tabgen_bench.preprocessing --datasets {name}")
        stat = path.stat()
        files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    digests = prepared_dataframe_digests(name, data_root)
    if expected["digest_label"] != DIGEST_LABEL:
        target.unlink(missing_ok=True)
        raise PreparedMismatchError(
            f"{name}: the TGB digests use {expected['digest_label']!r}, this "
            f"version of tabgen_bench computes {DIGEST_LABEL!r}")
    # Link 6: every split's digest must be the recorded one.
    differing = [split for split in SPLITS if digests.get(split) != expected["digests"][split]]
    if differing:
        target.unlink(missing_ok=True)
        raise PreparedMismatchError(
            f"{name}: the prepared {differing} split(s) do not have the TGB content "
            f"digest, so this is not the benchmark's data.  The TGB digests were "
            f"recorded with {expected.get('environment')}")
    # The evidence, for the next call to reuse.
    report = {
        "table": name,
        "verified_at": pd.Timestamp(started, unit="ns", tz="UTC").isoformat(timespec="seconds"),
        "verified_at_ns": started,
        "record_sha256": hashlib.sha256(section_text(expected).encode()).hexdigest(),
        "digests": dict(digests),
        "matches_tgb": True,
        "files": files,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1) + "\n")
    return {"digests": dict(digests), "matches_tgb": True, "reused": False}


def verify_tables(tables=None, *, data_root: Path | None = None,
                  prepared: bool = False, recheck: bool = False) -> dict:
    """Check tables already on disk against the benchmark's records.

    For each table (None: every table of the benchmark) the raw data must
    have the TGB raw schema and the recorded size and sha256 of every
    file.  With ``prepared=True`` the prepared splits must also have the
    TGB content digests.  Nothing is downloaded.  Returns
    ``{table: report}``; raises a :class:`ReproducibilityError` naming the
    table and what differed, or ``FileNotFoundError`` for data that is not
    there.

    A table whose raw files are untouched since they were last verified is
    not hashed again (its report says ``reused``; see :func:`verify_source`).
    ``recheck=True`` hashes every file regardless.
    """
    root = data_root or default_data_root()
    reports = {}
    for name in tables or benchmark_tables():
        report = verify_source(name, root, recheck=recheck)
        if report is None:
            raise KeyError(f"{name!r} is not a benchmark table")
        if prepared:
            report = {**report, "prepared": verify_prepared_dataframes(name, root, recheck=recheck)}
        reports[name] = report
    return reports


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="tables to verify (default: the benchmark's tables)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--prepared", action="store_true",
                        help="also check the prepared splits against the TGB "
                             "content digests")
    parser.add_argument("--recheck", action="store_true",
                        help="hash every raw file (and, with --prepared, digest every "
                             "prepared split) again, even one unchanged since it was "
                             "last checked")
    args = parser.parse_args()
    for name in args.datasets or benchmark_tables():
        report = verify_source(name, args.data_root, recheck=args.recheck)
        line = verification_line(name, report, args.data_root)
        if args.prepared:
            prepared = verify_prepared_dataframes(name, args.data_root, recheck=args.recheck)
            line += ("; prepared splits unchanged since they were checked" if prepared["reused"]
                     else "; prepared splits match the TGB digests")
        print(line, flush=True)


if __name__ == "__main__":
    main()
