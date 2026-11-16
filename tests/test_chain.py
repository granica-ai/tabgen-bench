"""The reproducibility chain, link by link, on the synthetic benchmark.

    TGB raw schema -> raw-data checksum -> stored row indices
        -> split-file checksum -> train / validation / test
        -> stored column specification -> content digests

Each test breaks exactly one link and expects the error that names it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tabgen_bench as tgb
from tabgen_bench import creator_tool, datasets, preprocessing, registry, validation

from conftest import DROPPED, DTYPES, NAME, ROWS, toy_frame, write_raw

SPLITS = ("train", "validation", "test")


def parquet_bytes(root, name=NAME):
    return {s: datasets.split_file(root, name, s).read_bytes() for s in SPLITS}


# ---------------------------------------------------------------- the whole chain

def test_tgb_build_holds_every_link(bench, tmp_path):
    root = bench.fresh_root(tmp_path, "tgb")
    spec = preprocessing.prepare_table(NAME, root)
    assert spec["split"] == "tgb" and spec["tgb"] is True
    assert spec["reproducibility"] == {
        "raw_schema_verified": True, "raw_checksum_verified": True,
        "split_checksum_verified": True, "column_spec_verified": True,
        "matches_tgb": True}
    assert spec["rows"] == {"train": 4000, "validation": 500, "test": 500}
    # the stored indices reproduce, byte for byte, the run they were recorded from
    assert parquet_bytes(root) == parquet_bytes(bench.root)
    assert validation.verify_prepared_dataframes(NAME, root)["matches_tgb"] is True


def test_tgb_build_draws_no_random_number(bench, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the TGB path asked for a random generator")
    monkeypatch.setattr(creator_tool, "_table_rng", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    root = bench.fresh_root(tmp_path, "norng")
    assert preprocessing.prepare_table(NAME, root)["tgb"] is True
    assert parquet_bytes(root) == parquet_bytes(bench.root)


def test_tgb_build_fits_nothing(bench, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the TGB path fitted something")
    monkeypatch.setattr(creator_tool, "_vocabulary", forbidden)
    monkeypatch.setattr(np, "median", forbidden)
    root = bench.fresh_root(tmp_path, "nofit")
    spec = preprocessing.prepare_table(NAME, root)
    assert spec["reproducibility"]["column_spec_verified"] is True
    assert parquet_bytes(root) == parquet_bytes(bench.root)


def test_two_tgb_builds_are_identical(bench, tmp_path):
    a, b = bench.fresh_root(tmp_path, "a"), bench.fresh_root(tmp_path, "b")
    preprocessing.prepare_table(NAME, a)
    preprocessing.prepare_table(NAME, b)
    assert parquet_bytes(a) == parquet_bytes(b)
    assert datasets.read_spec(NAME, a)["digests"] == datasets.read_spec(NAME, b)["digests"]


def test_splits_are_the_stored_positions_of_the_raw_table(bench, tmp_path):
    root = bench.fresh_root(tmp_path, "positions")
    preprocessing.prepare_table(NAME, root)
    record, positions = validation.load_split_indices(NAME)
    raw = datasets.read_raw(NAME, bench.raw)[0]
    for split in SPLITS:
        prepared = pd.read_parquet(datasets.split_file(root, NAME, split))
        assert np.all(np.diff(positions[split]) > 0)
        # num_a has no nulls and is not transformed: a direct view of the raw rows
        assert np.array_equal(prepared["num_a"].to_numpy(),
                              raw["num_a"].to_numpy()[positions[split]])
    assert record["rows"] == {"train": 4000, "validation": 500, "test": 500}


# ---------------------------------------------------------------- 1. raw schema

def test_schema_link_names_a_renamed_column(bench):
    frame = pd.read_parquet(bench.table_file).rename(columns={"count": "counts"})
    frame.to_parquet(bench.table_file, index=False)
    with pytest.raises(validation.SchemaMismatchError, match="count"):
        validation.verify_raw_schema(NAME, bench.root)
    # the build stops at this link, before the checksum, and writes nothing
    with pytest.raises(validation.SchemaMismatchError):
        preprocessing.prepare_table(NAME, bench.root)


def test_schema_link_catches_reordered_columns(bench):
    frame = pd.read_parquet(bench.table_file)
    frame[["num_missing", "num_a"] + list(frame.columns[2:])].to_parquet(
        bench.table_file, index=False)
    with pytest.raises(validation.SchemaMismatchError, match="position 0"):
        validation.verify_raw_schema(NAME, bench.root)


def test_schema_link_catches_a_row_count_change(bench):
    pd.read_parquet(bench.table_file).iloc[:-1].to_parquet(bench.table_file, index=False)
    with pytest.raises(validation.SchemaMismatchError, match=f"{ROWS - 1} rows"):
        validation.verify_raw_schema(NAME, bench.root)


def test_schema_link_catches_a_redeclared_dtype(bench):
    write_raw(bench.raw, toy_frame(), dtypes={**DTYPES, "count": "category"})
    with pytest.raises(validation.SchemaMismatchError, match="'count' is declared 'category'"):
        validation.verify_raw_schema(NAME, bench.root)


# ---------------------------------------------------------------- 2. raw checksum

def test_checksum_link_catches_one_changed_value(bench):
    assert validation.verify_source(NAME, bench.root)["all_files_match_record"] is True
    assert validation.verified_path(NAME, bench.root).exists()
    frame = pd.read_parquet(bench.table_file)
    frame.loc[3, "num_a"] += 1.0          # same schema, same row count
    frame.to_parquet(bench.table_file, index=False)
    assert validation.verify_raw_schema(NAME, bench.root)["matches_record"] is True
    with pytest.raises(validation.ChecksumMismatchError, match="dataset.parquet"):
        validation.verify_source(NAME, bench.root)
    # evidence about a download that no longer matches is removed
    assert not validation.verified_path(NAME, bench.root).exists()
    with pytest.raises(validation.ChecksumMismatchError):
        preprocessing.prepare_table(NAME, bench.root)


def _age_raw_files(bench, seconds=60):
    """Put the raw files' modification time in the past, as a real download's is."""
    import os
    import time
    then = time.time() - seconds
    for path in (bench.raw / NAME).rglob("*"):
        if path.is_file():
            os.utime(path, (then, then))


def test_verification_is_skipped_while_the_files_are_untouched(bench, monkeypatch):
    import os

    _age_raw_files(bench)
    first = validation.verify_source(NAME, bench.root, recheck=True)
    stored = json.loads(validation.verified_path(NAME, bench.root).read_text())
    assert first["reused"] is False and "reused" not in stored
    assert isinstance(stored["verified_at_ns"], int) and stored["verified_at"].endswith("+00:00")
    assert stored["record_sha256"] == validation.record_digest(registry.read_source(NAME))
    assert all(entry["mtime_ns"] < stored["verified_at_ns"] for entry in stored["files"])

    # untouched files: the stored report comes back and nothing is hashed
    hashed = []
    real_hash = validation.file_sha256
    monkeypatch.setattr(validation, "file_sha256",
                        lambda path: hashed.append(path) or real_hash(path))
    again = validation.verify_source(NAME, bench.root)
    assert again == {**stored, "reused": True} and hashed == []
    assert "unchanged since they were verified at" in validation.verification_line(
        NAME, again, bench.root)
    # a build skips it too; what it still hashes are the package's own records
    assert preprocessing.prepare_table(NAME, bench.root)["tgb"]
    assert not [path for path in hashed if bench.raw in Path(path).parents]
    hashed.clear()
    # recheck hashes regardless
    assert validation.verify_source(NAME, bench.root, recheck=True)["reused"] is False
    assert len(hashed) == len(stored["files"])

    # a file modified after the verification: verified again, from the files
    hashed.clear()
    os.utime(bench.table_file)                       # same bytes, a later time
    touched = validation.verify_source(NAME, bench.root)
    assert touched["reused"] is False and len(hashed) == len(stored["files"])

    # ... and when its content changed, the change is caught
    _age_raw_files(bench)
    validation.verify_source(NAME, bench.root)
    frame = pd.read_parquet(bench.table_file)
    frame.loc[3, "num_a"] += 1.0
    frame.to_parquet(bench.table_file, index=False)
    with pytest.raises(validation.ChecksumMismatchError, match="dataset.parquet"):
        validation.verify_source(NAME, bench.root)
    assert not validation.verified_path(NAME, bench.root).exists()


def test_a_stored_verification_is_not_reused_when_it_no_longer_applies(bench, tmp_path):
    import os

    source = registry.read_source(NAME)
    target = validation.verified_path(NAME, bench.root)

    def stored_after_verifying():
        _age_raw_files(bench)
        validation.verify_source(NAME, bench.root, recheck=True)
        assert validation.reusable_verification(NAME, bench.root, source) is not None
        return json.loads(target.read_text())

    # a file as recent as the verification itself is not trusted
    stored = stored_after_verifying()
    os.utime(bench.table_file, ns=(stored["verified_at_ns"], stored["verified_at_ns"]))
    stored["files"] = [dict(e, mtime_ns=stored["verified_at_ns"])
                       if e["path"].endswith(registry.DOWNLOAD_FILES["table"]) else e
                       for e in stored["files"]]
    target.write_text(json.dumps(stored))
    assert validation.reusable_verification(NAME, bench.root, source) is None
    # a file put back with another time (a copy, a restore)
    stored = stored_after_verifying()
    os.utime(bench.table_file, (1, 1))
    assert validation.reusable_verification(NAME, bench.root, source) is None
    # a report made against another record: a new pin, file list or schema
    stored = stored_after_verifying()
    changed = {**source, "origin": {**source["origin"], "revision": "1" * 40}}
    assert validation.reusable_verification(NAME, bench.root, changed) is None
    # a report from before the times were recorded, an unreadable one, none at all
    target.write_text(json.dumps({k: v for k, v in stored.items() if k != "verified_at_ns"}))
    assert validation.reusable_verification(NAME, bench.root, source) is None
    target.write_text("{")
    assert validation.reusable_verification(NAME, bench.root, source) is None
    target.unlink()
    assert validation.reusable_verification(NAME, bench.root, source) is None
    # a missing file falls through to the full check, which names it
    stored_after_verifying()
    bench.table_file.unlink()
    with pytest.raises(FileNotFoundError, match="is missing"):
        validation.verify_source(NAME, bench.root)
    # another data root over the same raw files has no report of its own
    assert validation.reusable_verification(
        NAME, bench.fresh_root(tmp_path, "other"), source) is None


def test_checksum_link_covers_the_metadata_files(bench):
    path = bench.raw / NAME / bench.table_file.parent.name / registry.DOWNLOAD_FILES["task"]
    path.write_text(path.read_text().replace("label", "labex"))
    with pytest.raises(validation.ChecksumMismatchError, match="task_metadata"):
        validation.verify_source(NAME, bench.root)


def test_a_missing_raw_file_is_reported_as_missing(bench):
    bench.table_file.unlink()
    with pytest.raises(FileNotFoundError, match="dataset.parquet"):
        validation.verify_source(NAME, bench.root)


# ---------------------------------------------------------------- 3-4. stored split

def rewrite_indices(bench, positions, *, update_record=True, **record_changes):
    path = registry.split_indices_path(NAME)
    creator_tool.write_split_indices(path, positions)
    record = json.loads((bench.record_dir / "tgb_split.json").read_text())
    if update_record:
        record["sha256"] = registry.file_sha256(path)
        record["rows"] = {s: int(len(positions[s])) for s in SPLITS}
    record.update(record_changes)
    (bench.record_dir / "tgb_split.json").write_text(json.dumps(record))


def test_split_file_checksum_catches_a_different_split(bench):
    _, positions = validation.load_split_indices(NAME)
    swapped = {"train": positions["train"], "validation": positions["test"],
               "test": positions["validation"]}       # a valid partition, but not ours
    rewrite_indices(bench, swapped, update_record=False)
    with pytest.raises(validation.SplitIndicesError, match="sha256"):
        validation.load_split_indices(NAME)
    with pytest.raises(validation.SplitIndicesError):
        preprocessing.prepare_table(NAME, bench.root)


def test_stored_split_must_put_every_row_in_exactly_one_split(bench):
    _, positions = validation.load_split_indices(NAME)
    duplicated = dict(positions, test=np.append(positions["test"][:-1], positions["train"][0]))
    rewrite_indices(bench, duplicated)
    with pytest.raises(validation.SplitIndicesError, match="more than one split"):
        validation.load_split_indices(NAME)


def test_stored_split_must_cover_the_whole_raw_table(bench):
    _, positions = validation.load_split_indices(NAME)
    rewrite_indices(bench, dict(positions, test=positions["test"][:-1]))
    with pytest.raises(validation.SplitIndicesError, match="5000 expected for a raw table"):
        validation.load_split_indices(NAME)


def test_stored_split_must_refer_to_the_pinned_raw_table(bench):
    _, positions = validation.load_split_indices(NAME)
    rewrite_indices(bench, positions, raw_sha256="0" * 64)
    with pytest.raises(validation.SplitIndicesError, match="raw table with sha256"):
        validation.load_split_indices(NAME)


def test_stored_split_counts_must_match_its_record(bench):
    _, positions = validation.load_split_indices(NAME)
    rewrite_indices(bench, positions, rows={"train": 3999, "validation": 501, "test": 500})
    with pytest.raises(validation.SplitIndicesError, match="tgb_split.json records"):
        validation.load_split_indices(NAME)


def test_split_indices_file_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    order = rng.permutation(1000)
    positions = {"train": np.sort(order[:800]), "validation": np.sort(order[800:900]),
                 "test": np.sort(order[900:])}
    path = creator_tool.write_split_indices(tmp_path / "s.parquet", positions)
    back = registry.read_split_indices_file(path)
    assert all(np.array_equal(back[s], positions[s]) for s in SPLITS)
    again = creator_tool.write_split_indices(tmp_path / "t.parquet", positions)
    assert path.read_bytes() == again.read_bytes()


# ---------------------------------------------------------------- 5. stored column spec

def reversed_ties(all_values, max_categories):
    """``_vocabulary`` as a different fit: folding at 256, equal counts ordered the other way.

    The synthetic table is under the TGB limit, so its fit ranks nothing;
    this stand-in is any code that would fit the vocabularies differently.
    """
    max_categories = 256
    order = preprocessing._text_order
    has_missing = bool(all_values.isna().any())
    distinct = sorted(set(all_values.dropna().astype(str)), key=order)
    reserved = [preprocessing.MISSING] if has_missing else []
    if max_categories and len(distinct) + len(reserved) > max_categories:
        budget = max(1, max_categories - 1 - len(reserved))
        counts = all_values.dropna().astype(str).value_counts(sort=False)
        ranked = sorted(counts.index, key=lambda label: (-int(counts[label]), order(label)),
                        reverse=False)
        tied = sorted(counts.index, key=lambda label: (-int(counts[label]),
                                                       tuple(-ord(c) for c in label)))
        assert ranked != tied
        return tuple(sorted(tied[:budget], key=order) + [preprocessing.OTHER] + reserved)
    return tuple(distinct + reserved)


def test_column_spec_is_what_makes_the_build_library_independent(bench, tmp_path, monkeypatch):
    spec = datasets.read_spec(NAME, bench.root)
    code = next(c for c in spec["columns"] if c["name"] == "code")
    assert len(code["categories"]) == 380 and "(other)" not in code["categories"]   # no limit
    monkeypatch.setattr(creator_tool, "_vocabulary", reversed_ties)
    # code that fits the vocabularies differently fits a different specification ...
    loaded = preprocessing.load_raw_splits(NAME, bench.root)
    held = pd.concat(loaded["splits"].values(), ignore_index=True)
    refit = creator_tool.fit_column_spec(NAME, held, loaded["splits"], loaded["dtypes"],
                                         loaded["task"], drop=DROPPED[NAME])
    other = next(c for c in refit["columns"] if c["name"] == "code")["categories"]
    assert other != code["categories"] and "(other)" in other
    # ... but a build applies the stored one, so it is the same TGB table
    root = bench.fresh_root(tmp_path, "tgb")
    assert preprocessing.prepare_table(NAME, root)["reproducibility"]["matches_tgb"]
    assert parquet_bytes(root) == parquet_bytes(bench.root)


def test_a_build_writes_its_splits_and_spec_and_nothing_else(bench, tmp_path):
    # the split is the package's tgb_split_indices.parquet; no copy of it is written
    root = bench.fresh_root(tmp_path, "tgb")
    spec = preprocessing.prepare_table(NAME, root)
    folder = root / "prepared" / NAME
    assert sorted(p.name for p in folder.iterdir()) == [
        "spec.json", "test.parquet", "train.parquet", "validation.parquet"]
    stored, _ = validation.load_split_indices(NAME)
    assert spec["split_record"]["split"] == "tgb"
    assert spec["split_record"]["indices_file"] == registry.TGB_SPLIT_INDICES
    assert spec["split_record"]["indices_sha256"] == stored["sha256"]
    assert spec["split_record"]["rule"] == stored["rule"]
    assert spec["split_record"]["rule"]["seed"] == 42 and spec["split"] == "tgb"
def test_column_spec_link_catches_an_edited_vocabulary(bench, tmp_path):
    path = bench.record_dir / "tgb_columns.json"
    spec = json.loads(path.read_text())
    code = next(c for c in spec["columns"] if c["name"] == "code")
    code["categories"] = code["categories"][1:]
    path.write_text(json.dumps(spec, indent=1) + "\n")
    with pytest.raises(validation.ColumnSpecError, match="sha256"):
        validation.load_column_spec(NAME)
    with pytest.raises(validation.ColumnSpecError):
        preprocessing.prepare_table(NAME, bench.fresh_root(tmp_path, "edited"))


def test_column_spec_must_read_columns_the_raw_schema_has(bench):
    path = bench.record_dir / "tgb_columns.json"
    spec = json.loads(path.read_text())
    next(c for c in spec["columns"] if "indicator_of" in c)["indicator_of"] = "no_such_column"
    path.write_text(registry.section_text(spec))
    record = json.loads((bench.record_dir / "tgb_prepared.json").read_text())
    record["columns_sha256"] = registry.file_sha256(path)
    (bench.record_dir / "tgb_prepared.json").write_text(json.dumps(record))
    with pytest.raises(validation.ColumnSpecError, match="no_such_column"):
        validation.load_column_spec(NAME)


def test_column_spec_records_how_every_column_is_made(bench):
    spec = validation.load_column_spec(NAME)
    by_name = {c["name"]: c for c in spec["columns"]}
    assert spec["column_order"] == ["num_a", "num_missing", "num_missing__missing", "count",
                                    "color", "note", "when", "stamp__days", "stamp__hours",
                                    "stamp__missing", "flag_const", "void", "code", "score", "label"]
    assert by_name["num_missing"]["source"] == "numerical"
    assert isinstance(by_name["num_missing"]["fill"], float)
    assert by_name["when"]["source"] == "datetime"
    assert by_name["num_missing__missing"]["indicator_of"] == "num_missing"
    assert by_name["color"]["categories"][-1] == "(missing)"
    # a constant column is kept, with the constant as its fill
    assert by_name["flag_const"]["kind"] == "numerical" and by_name["flag_const"]["fill"] == 1.0
    assert "flag_const" not in spec["dropped"] and "memo" in spec["dropped"]
    # an all-null column is a one-level categorical, whatever it was declared
    assert by_name["void"] == {"name": "void", "kind": "categorical", "source": "numerical",
                               "all_null": True, "categories": ["(missing)"],
                               "description": by_name["void"]["description"]}
    assert by_name["void"]["description"].endswith("missing in every row")
    assert spec["dropped"] == {"memo": spec["dropped"]["memo"]}


def test_the_listed_columns_are_dropped_and_the_creator_tool_finds_the_list(bench, tmp_path, monkeypatch):
    spec = validation.load_column_spec(NAME)
    # "memo" is listed, with its reason; "code" (380 labels in 4000 training rows) is not
    assert "memo" not in spec["column_order"] and spec["dropped"] == DROPPED[NAME]
    assert "code" in spec["column_order"]
    assert "memo" not in tgb.load_table(NAME, "train", data_root=bench.root).columns
    # the list is what the creator's tool computes from the data and the TGB split
    assert creator_tool.find_dropped_columns([NAME], bench.root) == DROPPED
    raw = tgb.load_raw(NAME, data_root=bench.root)
    _, positions = validation.load_split_indices(NAME)
    train, dtypes = raw.iloc[positions["train"]], {**DTYPES, "score": "float64"}
    assert creator_tool.high_cardinality_columns(train, dtypes, "label") == DROPPED[NAME]
    # a stricter ratio would also take "code", none at all nothing; the target is spared
    assert "code" in creator_tool.high_cardinality_columns(train, dtypes, "label", 0.01)
    assert creator_tool.high_cardinality_columns(train, dtypes, "label", 0) == {}
    assert "label" not in creator_tool.high_cardinality_columns(train, dtypes, "label", 0.0001)
    # the creator's fit drops exactly the list and keeps "code"
    fitted = creator_tool.record_columns(NAME, bench.fresh_root(tmp_path, "fitted"))
    assert fitted["dropped"] == DROPPED[NAME] and "code" in fitted["column_order"]
    # a list naming a column the table lacks, or the target, is refused
    for bad, message in (("no_such_column", "does not have"), ("label", "names the target")):
        monkeypatch.setattr(registry, "DROPPED_COLUMNS", {NAME: {bad: "x"}})
        with pytest.raises(ValueError, match=message):
            creator_tool.record_columns(NAME, bench.fresh_root(tmp_path, bad))


def test_a_timestamp_becomes_days_and_hours_and_a_date_stays_one_column(bench):
    spec = validation.load_column_spec(NAME)
    by_name = {c["name"]: c for c in spec["columns"]}
    # "when" has no time of day: a date, one numerical column of days, as before
    assert by_name["when"]["source"] == "datetime" and "part_of" not in by_name["when"]
    assert "stamp" not in by_name
    days, hours = by_name["stamp__days"], by_name["stamp__hours"]
    assert (days["kind"], days["source"], days["part_of"]) == ("numerical", "datetime_days", "stamp")
    assert (hours["kind"], hours["source"], hours["part_of"]) == ("numerical", "datetime_hours", "stamp")
    assert by_name["stamp__missing"]["indicator_of"] == "stamp"
    assert days["fill"] == float(int(days["fill"])) and 0 <= hours["fill"] < 24

    raw = tgb.load_raw(NAME, data_root=bench.root)
    _, positions = validation.load_split_indices(NAME)
    for split in SPLITS:
        made = tgb.load_table(NAME, split, data_root=bench.root)
        stamp = raw["stamp"].iloc[positions[split]].reset_index(drop=True)
        present = stamp.notna().to_numpy()
        assert list(made["stamp__missing"][present].unique()) == ["present"]
        assert set(made["stamp__missing"][~present]) <= {"missing"}
        d, h = made["stamp__days"].to_numpy(), made["stamp__hours"].to_numpy()
        assert np.array_equal(d, np.round(d)) and h.min() >= 0 and h.max() < 24
        # the two parts are the timestamp: date plus time of day, to the second
        rebuilt = (pd.Timestamp("1970-01-01") + pd.to_timedelta(d[present], "D")
                   + pd.to_timedelta(np.round(h[present] * 3600), "s"))
        assert np.array_equal(rebuilt.to_numpy(), stamp[present].to_numpy())
        # a missing timestamp takes the training median of each part
        assert set(d[~present]) <= {days["fill"]} and set(h[~present]) <= {hours["fill"]}


def test_which_datetime_columns_are_timestamps_is_judged_on_the_whole_table():
    midnight = pd.to_datetime(["2024-01-03", "2024-01-04", None, "2024-01-06"])
    frame = pd.DataFrame({
        "date": midnight,
        "one_clock_value": midnight.to_series(index=range(4)).where(
            lambda s: s != pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-06 00:00:01")),
        "empty": pd.Series([pd.NaT] * 4, dtype="datetime64[ns]"),
        "year": pd.period_range("2019", periods=4, freq="Y"),
        "number": [1.5, 2.5, 3.5, 4.5],
    })
    dtypes = {"date": "datetime64[ns]", "one_clock_value": "datetime64[ns]",
              "empty": "datetime64[ns]", "year": "period[Y-DEC]", "number": "float64"}
    assert creator_tool.timestamp_columns(frame, dtypes) == ("one_clock_value",)
    series = pd.Series(pd.to_datetime(["2024-01-03 13:30:00", "1969-12-31 23:59:59", None]))
    assert np.array_equal(preprocessing._whole_days(series), [19725.0, -1.0, np.nan],
                          equal_nan=True)
    assert np.allclose(preprocessing._hours(series), [13.5, 24 - 1 / 3600, np.nan],
                       equal_nan=True, rtol=0, atol=1e-12)
    for unit in ("s", "ms", "us"):
        other = series.astype(f"datetime64[{unit}]")
        assert np.array_equal(preprocessing._hours(other), preprocessing._hours(series),
                              equal_nan=True)


def test_cardinality_is_judged_on_the_rows_given_and_spares_the_target():
    frame = pd.DataFrame({
        "x": np.arange(100.0),
        "y": pd.Series([f"class {i}" for i in range(100)], dtype="string"),      # the target
        "text": pd.Series([f"t{i}" for i in range(100)], dtype="string"),
        "few": pd.Series(["a", "b"] * 50, dtype="string"),
        "eleven": pd.Series([f"e{i % 11}" for i in range(100)], dtype="string"),
        "ten": pd.Series([f"e{i % 10}" for i in range(100)], dtype="string"),
        "sparse": pd.Series([f"s{i}" if i < 11 else None for i in range(100)], dtype="string"),
    })
    dtypes = {"x": "float64", **{c: "string" for c in frame.columns[1:]}}
    high = creator_tool.high_cardinality_columns(frame, dtypes, "y")
    # 11 labels in 100 rows is above a tenth, 10 is not; missing rows count as rows
    assert set(high) == {"text", "eleven", "sparse"}
    assert high["eleven"] == "high cardinality: 11 labels in 100 training rows, ratio 0.110 above 0.1"
    assert creator_tool.high_cardinality_columns(frame, dtypes, "y", 0) == {}
    assert set(creator_tool.high_cardinality_columns(frame, dtypes, "few", 0.5)) == {"y", "text"}


def test_download_tables_checks_names_first_and_fetches_only_what_fails(bench, tmp_path, monkeypatch, capsys):
    import huggingface_hub
    import shutil

    from tabgen_bench import download

    good = tmp_path / "good"                       # a pristine copy to "download" from
    shutil.copytree(bench.raw / NAME, good / NAME)
    asked = []

    def fake_snapshot(**kwargs):
        asked.append(kwargs)
        for pattern in kwargs["allow_patterns"]:
            name = pattern.split("/")[0]
            assert name == NAME, "only benchmark tables are ever fetched"
            for path in (good / name).rglob("*"):
                if path.is_file():
                    target = bench.raw / path.relative_to(good)
                    assert not target.exists(), "a failed file is removed before the fetch"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)

    # names are checked before anything is fetched; a known, intact table is verified only
    reports = download.download_tables(["nothing_like_it", NAME], data_root=bench.root)
    out = capsys.readouterr().out
    assert asked == [] and reports["nothing_like_it"] is None
    assert reports[NAME]["reused"] is False and "nothing_like_it: not a table of this benchmark" in out
    assert out.index("nothing_like_it") < out.index(f"{NAME}:")

    # a file missing, as after an interrupted download: fetched again, then verified
    _age_raw_files(bench)
    (bench.table_file.parent / registry.DOWNLOAD_FILES["dtypes"]).unlink()
    with pytest.raises(FileNotFoundError):
        validation.verify_source(NAME, bench.root)
    reports = download.download_tables([NAME], data_root=bench.root)
    assert len(asked) == 1 and asked[0]["allow_patterns"] == [f"{NAME}/**"]
    assert asked[0]["revision"] == registry.read_source(NAME)["origin"]["revision"]
    assert reports[NAME]["reused"] is False and "is missing" in capsys.readouterr().out
    assert not list(bench.table_file.parent.glob("*.metadata"))

    # a file whose content differs from the record: removed, fetched again, verified
    _age_raw_files(bench)
    download.download_tables([NAME], data_root=bench.root)
    assert len(asked) == 1                          # intact now: nothing fetched
    frame = pd.read_parquet(bench.table_file)
    frame.loc[3, "num_a"] += 1.0
    frame.to_parquet(bench.table_file, index=False)
    reports = download.download_tables([NAME], data_root=bench.root)
    assert len(asked) == 2 and reports[NAME]["all_files_match_record"] is True
    assert "does not match the sha256" in capsys.readouterr().out
    assert pd.read_parquet(bench.table_file).loc[3, "num_a"] == frame.loc[3, "num_a"] - 1.0

    # the fetch brings back a file that still does not match: the error stands
    frame.to_parquet(good / bench.table_file.relative_to(bench.raw), index=False)
    frame.to_parquet(bench.table_file, index=False)
    with pytest.raises(validation.ChecksumMismatchError, match="dataset.parquet"):
        download.download_tables([NAME], data_root=bench.root)
    assert len(asked) == 3 and not validation.verified_path(NAME, bench.root).exists()


def test_download_skips_a_name_without_a_record_and_pins_the_rest(bench, tmp_path, monkeypatch, capsys):
    import huggingface_hub

    from tabgen_bench import download

    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: asked.append(kwargs))
    raw = download.download(["nothing_like_it", NAME], tmp_path / "empty")
    assert raw == tmp_path / "empty" / "raw" and raw.is_dir()
    assert [call["allow_patterns"] for call in asked] == [[f"{NAME}/**"]]
    assert asked[0]["revision"] == registry.read_source(NAME)["origin"]["revision"]
    assert asked[0]["ignore_patterns"] == ["*text_cache*"] and asked[0]["repo_type"] == "dataset"
    assert "nothing_like_it: not a table of this benchmark, skipped" in capsys.readouterr().out
    # nothing at all is asked for a list with no benchmark table in it
    assert download.download(["nothing_like_it"], tmp_path / "empty2") == tmp_path / "empty2" / "raw"
    assert len(asked) == 1


def test_verification_line_has_a_form_for_each_outcome(bench):
    where = str(validation.verified_path(NAME, bench.root))
    assert validation.verification_line(NAME, None, bench.root) == f"{NAME}: no origin record, not verified"
    fresh = validation.verify_source(NAME, bench.root, recheck=True)
    assert validation.verification_line(NAME, fresh, bench.root) == (
        f"{NAME}: 6 files match tests/toy@000000000000 -> {where}")
    reused = {**fresh, "reused": True}
    assert validation.verification_line(NAME, reused, bench.root) == (
        f"{NAME}: 6 files unchanged since they were verified at {fresh['verified_at']}, "
        f"tests/toy@000000000000 -> {where}")


def test_recheck_hashes_again_from_the_public_function_and_the_command_line(bench, monkeypatch, capsys):
    import sys

    _age_raw_files(bench)
    validation.verify_source(NAME, bench.root, recheck=True)
    hashed = []
    real_hash = validation.file_sha256
    monkeypatch.setattr(validation, "file_sha256", lambda path: hashed.append(path) or real_hash(path))
    assert tgb.verify_tables(data_root=bench.root)[NAME]["reused"] is True and hashed == []
    assert tgb.verify_tables(data_root=bench.root, recheck=True)[NAME]["reused"] is False
    assert len(hashed) == 6
    hashed.clear()
    monkeypatch.setattr(sys, "argv", ["tabgen-bench verify", "--data-root", str(bench.root)])
    validation.main()
    assert hashed == [] and "unchanged since they were verified" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["tabgen-bench verify", "--data-root", str(bench.root), "--recheck"])
    validation.main()
    assert len(hashed) == 6 and "6 files match" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["tabgen-bench download", "--data-root", str(bench.root)])
    hashed.clear()
    from tabgen_bench import download
    download.main()
    assert hashed == [] and "unchanged since they were verified" in capsys.readouterr().out


def test_remove_recorded_files_touches_only_the_recorded_paths(bench):
    from tabgen_bench import download

    raw, source = bench.raw, registry.read_source(NAME)
    cache = raw / ".cache" / "huggingface" / "download" / source["files"][0]["path"]
    cache = cache.with_name(cache.name + ".metadata")
    cache.parent.mkdir(parents=True)
    cache.write_text("0" * 40)
    stranger = bench.table_file.parent / "notes.txt"
    stranger.write_text("mine")
    removed = download.remove_recorded_files(NAME, bench.root, source)
    assert len(removed) == len(source["files"]) + 1 and cache in removed
    assert not bench.table_file.exists() and stranger.read_text() == "mine"
    assert download.remove_recorded_files(NAME, bench.root, source) == []


def test_a_row_subset_holds_whole_groups_and_is_checked(bench, tmp_path, monkeypatch):
    raw = tgb.load_raw(NAME, data_root=bench.root)
    rule = {"column": "code", "groups": 100, "seed": 7}
    monkeypatch.setattr(registry, "ROW_SUBSETS", {NAME: rule})
    # the split stored for the whole table does not fit the rule: a build refuses it
    with pytest.raises(validation.SplitIndicesError, match="row subset"):
        preprocessing.prepare_table(NAME, bench.fresh_root(tmp_path, "mismatch"))
    # the creator draws the subset, then the split on its rows, and stores both
    stored = creator_tool.draw_split(NAME, bench.root, val_fraction=0.1, test_fraction=0.1,
                                     seed=42)
    sub = stored["row_subset"]
    assert sub["groups_kept"] == 100 and sub["groups_total"] == 380
    assert json.loads((bench.record_dir / "tgb_split.json").read_text()) == stored
    assert stored["raw_rows"] == len(raw) and sum(stored["rows"].values()) == sub["rows_kept"]
    positions, record = creator_tool.subset_rows(raw, NAME, rule)
    assert record == sub and len(positions) == sub["rows_kept"]
    assert sub["rng"].startswith("numpy default_rng(crc32('<table>/<column>')")
    # the columns and digests are recorded from a build on the new split ...
    rec = bench.fresh_root(tmp_path, "record")
    creator_tool.record_columns(NAME, rec)
    held = pd.concat([tgb.load_table(NAME, s, data_root=rec) for s in ("train", "validation", "test")])
    assert len(held) == sub["rows_kept"] < len(raw)
    kept = set(held["code"].astype(str))
    assert len(kept) == 100
    assert kept.isdisjoint(set(raw["code"].astype(str)) - kept)       # whole groups
    assert set(raw["code"].astype(str).to_numpy()[positions]) == kept
    # ... and every later build is held to them, subset included
    again = preprocessing.prepare_table(NAME, bench.fresh_root(tmp_path, "again"))
    assert again["tgb"] and again["reproducibility"]["matches_tgb"]
    assert again["row_subset"] == sub
    # the registry's rule must still be the one the split was drawn with
    monkeypatch.setattr(registry, "ROW_SUBSETS", {NAME: {**rule, "groups": 120}})
    with pytest.raises(validation.SplitIndicesError, match="ROW_SUBSETS"):
        preprocessing.prepare_table(NAME, bench.fresh_root(tmp_path, "changed_rule"))
    monkeypatch.setattr(registry, "ROW_SUBSETS", {NAME: rule})
    # a split that breaks a group open is refused, and so is one of the wrong size
    _, positions = validation.load_split_indices(NAME)
    broken = {s: np.array(p) for s, p in positions.items()}
    broken["train"] = broken["train"][:-1]
    with pytest.raises(validation.SplitIndicesError, match="whole groups"):
        preprocessing.check_row_subset(NAME, raw, broken, sub)
    codes = raw["code"].astype(str).to_numpy()
    held = np.concatenate(list(positions.values()))
    whole = {s: np.array(p) for s, p in positions.items()}
    gone = codes[whole["train"][0]]                       # drop one whole group
    whole = {s: p[codes[p] != gone] for s, p in whole.items()}
    with pytest.raises(validation.SplitIndicesError, match="records 100 groups"):
        preprocessing.check_row_subset(NAME, raw, whole, sub)
    assert len(held) == sub["rows_kept"]
    for bad, message in (({"column": "nope", "groups": 3, "seed": 1}, "does not have"),
                         ({"column": "code", "groups": 0, "seed": 1}, "asks for 0")):
        with pytest.raises(ValueError, match=message):
            creator_tool.subset_rows(raw, NAME, bad)


def test_the_creator_tool_judges_cardinality_on_the_training_split(bench, tmp_path):
    # an identifier with 445 labels: one per nine training rows (ratio 0.111) and
    # one label for every other row, so 0.089 of the whole table
    raw = tgb.load_raw(NAME, data_root=bench.root)
    _, positions = validation.load_split_indices(NAME)
    ids = np.zeros(len(raw), dtype=int)
    ids[positions["train"]] = np.arange(len(positions["train"])) // 9 + 1
    frame = raw.assign(ident=pd.Series([f"i{v}" for v in ids]).astype("category"))
    dtypes = {**DTYPES, "ident": "category"}
    train = frame.iloc[positions["train"]]
    assert train["ident"].nunique() / len(train) > 0.1 > frame["ident"].nunique() / len(frame)
    assert "ident" not in creator_tool.high_cardinality_columns(frame, dtypes, "label")
    assert "ident" in creator_tool.high_cardinality_columns(train, dtypes, "label")
    # the tool reads the raw table and the stored split, and finds it
    root = tmp_path / "ident"
    write_raw(root / "raw", frame, dtypes=dtypes)
    found = creator_tool.find_dropped_columns([NAME], root)
    assert found[NAME]["ident"].startswith("high cardinality: 445 labels in 4000 training rows, ratio 0.111")
    assert found[NAME]["memo"] == DROPPED[NAME]["memo"]
    literal = creator_tool.dropped_columns_literal(found)
    assert literal.startswith("DROPPED_COLUMNS = {\n") and '"ident":' in literal
    namespace = {}
    exec(literal, namespace)
    assert namespace["DROPPED_COLUMNS"] == found


def test_a_dtype_override_reads_a_declared_category_as_the_number_it_is(bench, tmp_path,
                                                                         monkeypatch):
    spec = validation.load_column_spec(NAME)
    score = next(c for c in spec["columns"] if c["name"] == "score")
    assert score["kind"] == "numerical" and score["source"] == "numerical"
    built = datasets.read_spec(NAME, bench.root)
    assert built["dtype_overrides"] == {"score": "float64"}
    train = tgb.load_table(NAME, "train", data_root=bench.root)
    # the raw table stays as declared; the override is applied by the preparation
    raw = tgb.load_raw(NAME, data_root=bench.root)
    assert str(raw["score"].dtype) == "category" and train["score"].dtype == np.float64
    recast, dtypes = preprocessing.apply_dtype_overrides(NAME, raw, {**DTYPES, "score": "category"})
    assert str(recast["score"].dtype) == "float64" and dtypes["score"] == "float64"
    assert str(raw["score"].dtype) == "category"                   # the input is not touched
    _, positions = validation.load_split_indices(NAME)
    assert np.array_equal(train["score"].to_numpy(), recast["score"].to_numpy()[positions["train"]])
    # the bundle's own declaration is untouched: it is what the raw schema records
    declared = {c["name"]: c["dtype"] for c in registry.read_source(NAME)["schema"]["columns"]}
    assert declared["score"] == "category"
    assert validation.verify_raw_schema(NAME, bench.root)["matches_record"] is True
    # without the override the column is a near-unique category: the creator's
    # tool would list it, and a build then carries it as a 4,000-label vocabulary
    monkeypatch.setattr(registry, "DTYPE_OVERRIDES", {})
    same, same_dtypes = preprocessing.apply_dtype_overrides(NAME, raw, DTYPES)
    assert same is raw and same_dtypes is DTYPES                   # no overrides: untouched
    found = creator_tool.find_dropped_columns([NAME], bench.root)[NAME]
    assert found["score"].startswith("high cardinality: 4000 labels in 4000 training rows")
    loaded = preprocessing.load_raw_splits(NAME, bench.root)
    plain = creator_tool.fit_column_spec(NAME, pd.concat(loaded["splits"].values(), ignore_index=True),
                                         loaded["splits"], loaded["dtypes"], loaded["task"])
    score = next(c for c in plain["columns"] if c["name"] == "score")
    assert score["kind"] == "categorical" and len(score["categories"]) > 3000
    monkeypatch.setattr(registry, "DTYPE_OVERRIDES", {NAME: {"no_such_column": "float64"}})
    tgb.load_raw(NAME, data_root=bench.root)                        # reading is unaffected
    with pytest.raises(KeyError, match="no_such_column"):
        preprocessing.prepare_table(NAME, bench.fresh_root(tmp_path, "unknown"))


def test_record_columns_keeps_stored_vocabularies_unless_asked_to_refit(bench, tmp_path):
    before = json.loads((bench.record_dir / "tgb_columns.json").read_text())
    code_before = next(c for c in before["columns"] if c["name"] == "code")["categories"]
    # a library that breaks ties differently refits, yet the stored vocabulary survives
    import unittest.mock as mock
    with mock.patch.object(creator_tool, "_vocabulary", reversed_ties):
        creator_tool.record_columns(NAME, bench.fresh_root(tmp_path, "refit"))
    after = json.loads((bench.record_dir / "tgb_columns.json").read_text())
    assert after == before
    assert next(c for c in after["columns"] if c["name"] == "code")["categories"] == code_before
    root = bench.fresh_root(tmp_path, "after")
    assert preprocessing.prepare_table(NAME, root)["reproducibility"]["matches_tgb"]
    assert parquet_bytes(root) == parquet_bytes(bench.root)
    # refit_vocabularies fits them again as well
    with mock.patch.object(creator_tool, "_vocabulary", reversed_ties):
        creator_tool.record_columns(NAME, bench.fresh_root(tmp_path, "vocab"),
                                    refit_vocabularies=True)
    refit = json.loads((bench.record_dir / "tgb_columns.json").read_text())
    code_after = next(c for c in refit["columns"] if c["name"] == "code")["categories"]
    assert len(code_after) == 256 and "(other)" in code_after
    root = bench.fresh_root(tmp_path, "refit_vocab")
    assert preprocessing.prepare_table(NAME, root)["reproducibility"]["matches_tgb"]
    # recording builds the table too, and the apply reproduces the fit
    built = creator_tool.record_columns(NAME, bench.fresh_root(tmp_path, "built"))
    assert built["reproducibility"]["matches_tgb"] is True


def test_tied_counts_are_ranked_by_label_not_by_row_order():
    rng = np.random.default_rng(3)
    labels = pd.Series([f"v{i}" for i in range(40)] * 2 + ["common"] * 5, dtype=object)
    first = creator_tool._vocabulary(labels, 9)
    for _ in range(5):
        shuffled = labels.iloc[rng.permutation(len(labels))].reset_index(drop=True)
        assert creator_tool._vocabulary(shuffled, 9) == first
    assert first == ("common", "v0", "v1", "v10", "v11", "v12", "v13", "v14", "v15", "(other)")
    # labels that are one number are still ranked one way, whatever order a set yields them in
    same = pd.Series(["01", "001", "1", "1.0", "2", "x", "nan"], dtype=object)
    assert creator_tool._vocabulary(same, 0) == ("001", "01", "1", "1.0", "2", "nan", "x")


# ---------------------------------------------------------------- 6. content digests

def test_digest_link_stops_a_build_that_is_not_the_benchmark(bench, tmp_path):
    record = json.loads((bench.record_dir / "tgb_prepared.json").read_text())
    record["digests"]["test"] = "0" * 64
    (bench.record_dir / "tgb_prepared.json").write_text(json.dumps(record))
    root = bench.fresh_root(tmp_path, "strict")
    with pytest.raises(validation.PreparedMismatchError, match=r"\['test'\]"):
        preprocessing.prepare_table(NAME, root)
    assert not datasets.spec_file(root, NAME).exists()      # nothing was written
    spec = preprocessing.prepare_table(NAME, root, strict=False)
    assert spec["tgb"] is True
    assert spec["reproducibility"]["matches_tgb"] is False


def test_prepared_files_are_held_to_the_digest_on_load(bench):
    assert set(tgb.load_table(NAME, data_root=bench.root)) == set(SPLITS)
    path = datasets.split_file(bench.root, NAME, "validation")
    frame = pd.read_parquet(path)
    frame.loc[0, "num_a"] += 1e-9
    frame.to_parquet(path, index=False)
    with pytest.raises(validation.PreparedMismatchError, match="validation"):
        tgb.load_table(NAME, data_root=bench.root)
    with pytest.raises(validation.PreparedMismatchError, match="validation"):
        tgb.load_table(NAME, "validation", data_root=bench.root)
    with pytest.raises(validation.PreparedMismatchError, match="validation"):
        tgb.load_table(NAME, "train", data_root=bench.root)  # the table is checked whole
    with pytest.raises(validation.PreparedMismatchError):
        validation.verify_prepared_dataframes(NAME, bench.root)


# ---------------------------------------------------------------- what is not TGB

def test_every_build_sits_on_the_stored_split(bench, tmp_path):
    spec = preprocessing.prepare_table(NAME, bench.fresh_root(tmp_path, "plain"))
    assert spec["split"] == "tgb" and spec["tgb"] is True
    assert spec["reproducibility"]["matches_tgb"] is True
    assert spec["max_categories"] == 15000 and spec["missing_indicator_min_frac"] == 0.01


def test_unverified_build_says_so(bench, tmp_path):
    spec = preprocessing.prepare_table(NAME, bench.fresh_root(tmp_path, "noverify"),
                                       verify=False)
    assert spec["reproducibility"]["raw_checksum_verified"] is False
    assert spec["reproducibility"]["split_checksum_verified"] is True
    assert spec["reproducibility"]["column_spec_verified"] is True


def test_loading_prepares_a_missing_table_and_checks_one_that_is_there(bench, tmp_path, monkeypatch):
    import huggingface_hub

    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: asked.append(kwargs))
    # an empty data root (over the raw files): the table is built on first use
    root = bench.fresh_root(tmp_path, "fresh")
    assert not (root / "prepared").exists()
    splits = tgb.load_table(NAME, data_root=root)
    assert set(splits) == set(SPLITS) and asked == []
    assert parquet_bytes(root) == parquet_bytes(bench.root)
    assert validation.prepared_verified_path(NAME, root).exists()
    # a second load checks the files once more (they are too fresh to trust) ...
    _age_raw_files(bench)
    digested = []
    real = validation.compute_dataframe_digest
    monkeypatch.setattr(validation, "compute_dataframe_digest", lambda f: digested.append(1) or real(f))
    tgb.load_table(NAME, "train", data_root=root)
    assert len(digested) == 3
    # ... and once the check is on record and the files are older than it, not again
    stored = json.loads(validation.prepared_verified_path(NAME, root).read_text())
    then = stored["verified_at_ns"] - 60 * 10**9
    for s in SPLITS:
        import os
        os.utime(datasets.split_file(root, NAME, s), ns=(then, then))
    stored["files"] = [{**e, "mtime_ns": then} for e in stored["files"]]
    validation.prepared_verified_path(NAME, root).write_text(json.dumps(stored))
    digested.clear()
    assert tgb.load_spec(NAME, data_root=root)["column_order"] == datasets.read_spec(NAME, bench.root)["column_order"]
    tgb.load_table(NAME, data_root=root)
    assert digested == []
    assert validation.verify_prepared_dataframes(NAME, root)["reused"] is True
    assert validation.verify_prepared_dataframes(NAME, root, recheck=True)["reused"] is False
    # a split file removed: the table is built again, not reported missing
    datasets.split_file(root, NAME, "test").unlink()
    assert len(tgb.load_table(NAME, "test", data_root=root)) == 500
    # a split file altered: the check fails on load and the report is gone
    path = datasets.split_file(root, NAME, "validation")
    frame = pd.read_parquet(path)
    frame.loc[0, "num_a"] += 1e-9
    frame.to_parquet(path, index=False)
    with pytest.raises(validation.PreparedMismatchError, match="validation"):
        tgb.load_spec(NAME, data_root=root)
    assert not validation.prepared_verified_path(NAME, root).exists()


def test_a_build_has_no_policy_or_split_to_choose(bench):
    for option in ({"split": "fractions"}, {"max_categories": 0}, {"record_tgb": True},
                   {"refit_columns": True}, {"missing_policy": "drop_rows"}):
        with pytest.raises(TypeError):
            preprocessing.prepare_table(NAME, bench.root, **option)


# ---------------------------------------------------------------- public interface

def test_public_functions_build_and_load_the_benchmark(bench, tmp_path):
    root = bench.fresh_root(tmp_path, "public")
    assert tgb.list_tables() == (NAME,)
    info = tgb.table_info(NAME)
    assert info["target"] == "label" and info["split_rows"]["train"] == 4000
    reports = tgb.verify_tables(data_root=root)
    assert reports[NAME]["schema"] == {"rows": ROWS, "columns": 13, "matches_record": True}
    manifest = tgb.prepare_tables(data_root=root)
    assert manifest["tables"][NAME]["tgb"] is True
    assert manifest["tables"][NAME]["matches_tgb"] is True
    splits = tgb.load_table(NAME, data_root=root)
    assert [len(splits[s]) for s in SPLITS] == [4000, 500, 500]
    spec = tgb.load_spec(NAME, data_root=root)
    assert list(splits["train"].columns) == spec["column_order"]
    assert "flag_const" in spec["column_order"] and "num_missing__missing" in spec["column_order"]
    assert set(splits["train"]["flag_const"]) == {1.0}
    assert set(splits["train"]["void"]) == {"(missing)"} and "void" in spec["column_order"]
    assert tgb.verify_tables(data_root=root, prepared=True)[NAME]["prepared"]["matches_tgb"]
    assert len(tgb.load_raw(NAME, data_root=root)) == ROWS
    with pytest.raises(KeyError):
        tgb.table_info("not_a_table")
    with pytest.raises(KeyError, match="not a benchmark table"):
        tgb.load_table("not_a_table", data_root=root)
    with pytest.raises(ValueError, match="split must be one of"):
        tgb.load_table(NAME, "dev", data_root=root)


def test_every_public_name_exists_and_errors_share_a_base():
    assert all(hasattr(tgb, name) for name in tgb.__all__)
    assert len(set(tgb.__all__)) == len(tgb.__all__)
    assert set(tgb._HOME) | {"__version__"} == set(tgb.__all__)
    assert set(tgb.__all__) <= set(dir(tgb))
    with pytest.raises(AttributeError):
        tgb.no_such_name
    for error in (tgb.SchemaMismatchError, tgb.ChecksumMismatchError,
                  tgb.SplitIndicesError, tgb.ColumnSpecError, tgb.PreparedMismatchError):
        assert issubclass(error, tgb.ReproducibilityError)
    assert issubclass(tgb.ReproducibilityError, ValueError)
