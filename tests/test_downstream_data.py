"""True-observation recovery, train-only vocabularies and checked split alignment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabgen_bench.downstream.creator_tool import DownstreamDataset, load_mask_creation_data


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    """Small prepared/raw fixture with a numeric null and no indicator for it."""
    from tabgen_bench import datasets, preprocessing, registry, validation

    raw_train = pd.DataFrame(
        {
            "rare_null": [1.0, np.nan, 3.0, np.inf],
            "color": ["red", "blue", None, "red"],
            "stamp": pd.to_datetime(["2020-01-01", "2020-02-01", None, "2020-03-01"]),
            "label": ["yes", "no", "yes", "no"],
        }
    )
    raw_test = pd.DataFrame(
        {
            "rare_null": [np.nan, 5.0, -np.inf],
            "color": ["test-only", None, "red"],
            "stamp": pd.to_datetime([None, "2021-01-01", "2021-02-01"]),
            "label": ["yes", "no", "yes"],
        }
    )
    columns = [
        {"name": "rare_null", "kind": "numerical", "source": "numerical", "fill": 2.0},
        {
            "name": "color",
            "kind": "categorical",
            "categories": ["red", "blue", "test-only", "(missing)"],
        },
        {
            "name": "stamp__days",
            "kind": "numerical",
            "source": "datetime_days",
            "part_of": "stamp",
            "fill": 0.0,
        },
        {
            "name": "stamp__hours",
            "kind": "numerical",
            "source": "datetime_hours",
            "part_of": "stamp",
            "fill": 0.0,
        },
        {
            "name": "stamp__missing",
            "kind": "categorical",
            "indicator_of": "stamp",
            "categories": ["present", "missing"],
        },
        {"name": "label", "kind": "categorical", "categories": ["yes", "no"]},
    ]
    spec = {
        "columns": columns,
        "column_order": [c["name"] for c in columns],
        "target": "label",
        "dropped": {},
    }
    splits = {"train": raw_train, "validation": raw_test.iloc[:1], "test": raw_test}
    prepared = preprocessing.encode_splits(spec, splits)
    root = tmp_path / "data"
    directory = root / "prepared" / "toy"
    directory.mkdir(parents=True)
    for split, frame in prepared.items():
        frame.to_parquet(directory / f"{split}.parquet", index=False)
    (directory / "spec.json").write_text(json.dumps(spec))
    positions = {
        "train": np.array([0, 2, 4, 6]),
        "validation": np.array([7]),
        "test": np.array([1, 3, 5]),
    }
    calls = []
    monkeypatch.setattr(registry, "benchmark_tables", lambda: ("toy",))
    monkeypatch.setattr(validation, "load_column_spec", lambda name: spec)
    monkeypatch.setattr(
        validation, "load_split_indices", lambda name: ({"sha256": "split-sha"}, positions)
    )

    def verify(name, data_root):
        calls.append(("prepared", name, data_root))
        return {"digests": {"train": "train-sha", "test": "test-sha"}}

    def load_raw(name, data_root, *, verify):
        calls.append(("raw", name, data_root, verify))
        return {
            "splits": splits,
            "task": {"target_column_name": "label", "problem_type": "binary_classification"},
        }

    monkeypatch.setattr(validation, "verify_prepared_dataframes", verify)
    monkeypatch.setattr(preprocessing, "load_raw_splits", load_raw)
    monkeypatch.setattr(
        datasets, "ensure_prepared", lambda *a, **kw: pytest.fail("must not prepare or download")
    )
    return root, splits, positions, calls


def test_loader_recovers_original_missingness_and_uses_training_vocabulary(local_data):
    root, _, positions, calls = local_data
    dataset = load_mask_creation_data("toy", root, max_rows=None)
    assert [call[0] for call in calls] == ["prepared", "raw"]
    assert calls[1][-1] is True
    assert dataset.observed_train[:, 0].tolist() == [True, False, True, False]
    assert dataset.observed_test[:, 0].tolist() == [False, True, False]
    assert dataset.observed_test[:, 1].tolist() == [True, False, True]
    assert not dataset.observed_test[0, 2:4].any()
    assert dataset.observed_test[:, 4].all()  # missingness indicators are known
    assert dataset.metadata["categories"]["color"] == ["blue", "red"]
    assert dataset.test.loc[0, "color"] == "test-only"
    assert dataset.feature_groups[2] == ("stamp__days", "stamp__hours", "stamp__missing")
    np.testing.assert_array_equal(dataset.test_row_ids, positions["test"])


def test_subsampling_preserves_raw_source_ids_and_changes_fingerprint(local_data):
    root, splits, positions, _ = local_data
    full = load_mask_creation_data("toy", root, max_rows=None)
    sample = load_mask_creation_data("toy", root, max_rows=2, seed=7)
    again = load_mask_creation_data("toy", root, max_rows=2, seed=7)
    np.testing.assert_array_equal(sample.test_row_ids, again.test_row_ids)
    assert sample.fingerprint == again.fingerprint != full.fingerprint
    for index, raw_id in enumerate(sample.test_row_ids):
        source_index = list(positions["test"]).index(raw_id)
        assert sample.test.loc[index, "label"] == splits["test"].loc[source_index, "label"]
        np.testing.assert_array_equal(sample.observed_test[index], full.observed_test[source_index])


def test_loader_refuses_missing_local_prepared_data_before_any_preparation(local_data):
    root, _, _, calls = local_data
    (root / "prepared" / "toy" / "test.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Local prepared benchmark"):
        load_mask_creation_data("toy", root)
    assert calls == []


def test_loader_rejects_split_row_misalignment(local_data):
    root, splits, _, _ = local_data
    splits["test"] = splits["test"].iloc[:-1]
    with pytest.raises(ValueError, match="raw split rows"):
        load_mask_creation_data("toy", root)


def test_dataset_copies_masks_and_validates_schema():
    frame = pd.DataFrame({"x": [1.0], "y": [2.0]})
    observed = np.ones(frame.shape, dtype=bool)
    dataset = DownstreamDataset(
        "toy",
        frame,
        frame,
        {"x": "numerical", "y": "numerical"},
        "y",
        observed,
        observed,
        np.array([2]),
        "fingerprint",
    )
    observed[:] = False
    frame.loc[0, "x"] = 99
    assert dataset.observed_test.all() and dataset.train.loc[0, "x"] == 1
    with pytest.raises(ValueError, match="read-only"):
        dataset.observed_test[0, 0] = False
    with pytest.raises(ValueError, match="partition"):
        DownstreamDataset(
            "toy",
            frame,
            frame,
            {"x": "numerical", "y": "numerical"},
            "y",
            observed,
            observed,
            np.array([2]),
            "fingerprint",
            feature_groups=(("x",),),
        )
