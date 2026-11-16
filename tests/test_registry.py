"""The shipped registry is self-consistent.  Needs no downloaded data.

For every table of the benchmark: the origin record is complete, the stored
split has its recorded sha256 and partitions the raw rows 80 / 10 / 10, and
the digests refer to that split.
"""

from __future__ import annotations

import pytest

from tabgen_bench import creator_tool, registry, validation

SPLITS = ("train", "validation", "test")


def test_the_list_and_the_records_agree():
    assert len(registry.TABLES) == 28
    assert list(registry.TABLES) == sorted(set(registry.TABLES))
    assert set(creator_tool.recorded_tables()) == set(registry.TABLES)
    assert registry.benchmark_tables() == registry.TABLES


@pytest.mark.parametrize("name", registry.TABLES)
def test_origin_record(name):
    source = registry.read_source(name)
    origin, files, dataset, schema = (source[s] for s in registry.SOURCE_SECTIONS)
    assert origin["table"] == name and len(origin["revision"]) == 40
    assert origin["revision"] in origin["url"]
    assert sorted(f["name"] for f in files) == sorted(registry.DOWNLOAD_FILES.values())
    assert all(len(f["sha256"]) == 64 and f["size"] > 0 for f in files)
    assert all(f["path"].startswith(origin["directory"] + "/") for f in files)
    assert schema["rows"] == dataset["rows"]
    assert len(schema["columns"]) == dataset["columns"]
    names = [c["name"] for c in schema["columns"]]
    assert len(set(names)) == len(names) and dataset["target"] in names


@pytest.mark.parametrize("name", registry.TABLES)
def test_stored_split(name):
    record, positions = validation.load_split_indices(name)   # sha256 + partition
    source = registry.read_source(name)
    total = source["schema"]["rows"]
    if record.get("row_subset"):                 # a whole-group subset of the raw rows
        total = record["row_subset"]["rows_kept"]
    assert sum(len(positions[s]) for s in SPLITS) == total
    assert abs(len(positions["train"]) - 0.8 * total) <= 2
    assert abs(len(positions["validation"]) - 0.1 * total) <= 2
    assert abs(len(positions["test"]) - 0.1 * total) <= 2
    assert record["rule"].get("cut_regime", "iid") == "iid" and record["rule"]["seed"] == 42


@pytest.mark.parametrize("name", registry.TABLES)
def test_stored_column_spec(name):
    spec = validation.load_column_spec(name)          # sha256 + raw columns exist
    source = registry.read_source(name)
    assert len(spec["column_order"]) == source["prepared"]["columns"]
    raw = {c["name"] for c in source["schema"]["columns"]}
    made_from = {c.get("indicator_of") or c.get("part_of") or c["name"]
                 for c in spec["columns"]}
    assert made_from | set(spec["dropped"]) == raw      # every raw column is accounted for
    for column in spec["columns"]:
        if column.get("all_null"):
            assert column["kind"] == "categorical" and column["categories"] == ["(missing)"]
            assert column["source"] in ("numerical", "datetime", "categorical")
            continue
        if column["kind"] == "categorical":
            assert len(column["categories"]) >= 2
            labels = [c for c in column["categories"] if c not in ("(other)", "(missing)")]
            assert len(labels) <= 15000
            assert ("(other)" in column["categories"]) == (len(labels) == 15000)
            assert len(set(column["categories"])) == len(column["categories"])
        else:
            assert column["source"] in ("numerical", "datetime", "datetime_days",
                                        "datetime_hours")
            assert column["fill"] == column["fill"]      # a number, not NaN
    assert source["dataset"]["target"] in spec["column_order"]


def test_the_near_unique_columns_are_dropped():
    expected = {
        "kickstarter": {"name", "blurb", "creator_name", "profile_blurb"},
        "sf_permit_time": {"Description"},
        "mercari_price_suggestion": {"item_description", "name"},
        "consumer_complaints": {"Consumer complaint narrative"},
        "lending_club": {"emp_title", "desc"},
        "amex_non_iid": {"customer_ID"},      # 0.083 of the table, 0.103 of the training split
    }
    found = {}
    for name in registry.TABLES:
        dropped = registry.read_source(name)["columns"]["dropped"]
        high = {column for column, why in dropped.items() if why.startswith("high cardinality")}
        if high:
            found[name] = high
    assert found == expected == {t: set(c) for t, c in registry.DROPPED_COLUMNS.items()}


def test_timestamps_are_split_into_days_and_hours():
    expected = {
        "kickstarter": {"created_at", "launched_at", "deadline"},
        "sf_permit_time": {"Filed Date"},
        "delivery_eta": {"timestamp"},
        "cooking_time": {"timestamp"},
        "climate_model_weather_forecasting": {"fact_time"},
    }
    found, dates = {}, 0
    for name in registry.TABLES:
        source = registry.read_source(name)
        columns = {c["name"]: c for c in source["columns"]["columns"]}
        parts = {}
        for column in columns.values():
            if "part_of" in column:
                parts.setdefault(column["part_of"], set()).add(column["source"])
                assert column["name"] in (f"{column['part_of']}__days",
                                          f"{column['part_of']}__hours")
                assert column["part_of"] not in columns          # the raw name is not kept
                if column["source"] == "datetime_hours":
                    assert 0 <= column["fill"] < 24
                else:
                    assert column["fill"] == int(column["fill"])
            elif column.get("source") == "datetime" and column["kind"] == "numerical":
                dates += 1
        if parts:
            assert all(v == {"datetime_days", "datetime_hours"} for v in parts.values()), name
            found[name] = set(parts)
        declared = {c["name"] for c in source["schema"]["columns"]
                    if c["dtype"].startswith(("datetime", "period"))}
        assert set(parts) <= declared
    assert found == expected
    assert dates == 11                # the date-only columns stay one column each
    # the one all-null datetime column is a (missing) level, neither a date nor a timestamp
    void = [c for c in registry.read_source("lending_club")["columns"]["columns"]
            if c["name"] == "sec_app_earliest_cr_line"]
    assert void and void[0]["all_null"] and void[0]["source"] == "datetime"


def test_row_subsets_are_recorded_and_are_whole_groups_of_the_stored_split():
    assert set(registry.ROW_SUBSETS) == {"sepsis_prediction"}
    for name in registry.TABLES:
        source = registry.read_source(name)
        rule = registry.row_subset(name)
        recorded = source["split"].get("row_subset")
        if rule is None:
            assert recorded is None
            assert sum(source["split"]["rows"].values()) == source["schema"]["rows"]
            continue
        assert recorded["column"] == rule["column"] and recorded["groups"] == rule["groups"]
        assert recorded["seed"] == rule["seed"]
        assert recorded["groups_kept"] == rule["groups"] < recorded["groups_total"]
        assert sum(source["split"]["rows"].values()) == recorded["rows_kept"] < source["schema"]["rows"]
        assert recorded["rows_total"] == source["schema"]["rows"]


def test_dropped_columns_are_raw_columns_and_are_recorded_as_dropped():
    assert {t: set(c) for t, c in registry.DROPPED_COLUMNS.items()} == {
        "kickstarter": {"name", "blurb", "creator_name", "profile_blurb"},
        "sf_permit_time": {"Description"},
        "mercari_price_suggestion": {"item_description", "name"},
        "consumer_complaints": {"Consumer complaint narrative"},
        "lending_club": {"emp_title", "desc"},
        "amex_non_iid": {"customer_ID"},      # 0.083 of the table, 0.103 of the training split
    }
    for name in registry.TABLES:
        source = registry.read_source(name)
        declared = {c["name"] for c in source["schema"]["columns"]}
        made = {c["name"] for c in source["columns"]["columns"]}
        listed = registry.dropped_columns(name)
        assert source["columns"]["dropped"] == listed
        for column, reason in listed.items():
            assert column in declared and column not in made and column != source["dataset"]["target"]
            assert reason.startswith("high cardinality:")
            assert f"in {source['split']['rows']['train']} training rows" in reason
            assert reason.endswith(f"above {creator_tool.MAX_CARDINALITY_RATIO:g}")


def test_dtype_overrides_name_real_columns_and_take_effect():
    assert registry.DTYPE_OVERRIDES == {
        "give_me_some_credit": {"RevolvingUtilizationOfUnsecuredLines": "float64"}}
    for name, overrides in registry.DTYPE_OVERRIDES.items():
        source = registry.read_source(name)
        declared = {c["name"]: c["dtype"] for c in source["schema"]["columns"]}
        made = {c["name"]: c for c in source["columns"]["columns"]}
        for column, dtype in overrides.items():
            assert declared[column] == "category" and dtype == "float64"
            assert made[column]["kind"] == "numerical" and made[column]["source"] == "numerical"
            assert column not in source["columns"]["dropped"]


@pytest.mark.parametrize("name", registry.TABLES)
def test_tgb_digests(name):
    source = registry.read_source(name)
    prepared, split = source["prepared"], source["split"]
    assert prepared["digest_label"] == validation.DIGEST_LABEL
    assert prepared["rows"] == split["rows"]
    assert all(len(prepared["digests"][s]) == 64 for s in SPLITS)
    assert len(set(prepared["digests"].values())) == 3
    assert prepared["policy"] == {"max_categories": 15000,
                                  "missing_indicator_min_frac": 0.01}
