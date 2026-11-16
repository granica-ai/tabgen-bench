"""Task consumers receive prepared values with complete groups moved together."""

from __future__ import annotations

import inspect
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tabgen_bench as tgb
from tabgen_bench.downstream import tasks
from tabgen_bench.downstream.creator_tool import DownstreamDataset
from tabgen_bench.downstream.masks import _CoordinateMask

from conftest import NAME


@pytest.fixture
def grouped(monkeypatch):
    frame = pd.DataFrame({
        "age": [20.0, 30.0, 40.0],
        "income": [10.0, 30.0, 50.0],  # row zero is a prepared fill
        "income__missing": ["missing", "present", "present"],
        "stamp__days": [2.0, 3.0, 4.0],
        "stamp__hours": [6.0, 8.0, 10.0],  # row zero is a prepared fill
        "stamp__missing": ["present", "present", "present"],
        "label": ["yes", "no", "(missing)"],
        "label__missing": ["present", "present", "missing"],
    })
    natural = np.ones(frame.shape, dtype=bool)
    natural[0, [1, 4]] = False
    natural[2, 6] = False
    groups = (("age",), ("income", "income__missing"),
              ("stamp__days", "stamp__hours", "stamp__missing"),
              ("label", "label__missing"))
    data = DownstreamDataset(
        "grouped", frame.iloc[:1], frame,
        {c: "numerical" if c in ("age", "income", "stamp__days", "stamp__hours")
         else "categorical" for c in frame},
        "label", natural[:1], natural, np.arange(3), "grouped-identity",
        metadata={"indicator_columns": ["income__missing", "stamp__missing", "label__missing"]},
        feature_groups=groups)
    prediction = natural.copy()
    prediction[:, 6:] = False
    imputation = natural.copy()
    imputation[0, 3:6] = False  # includes the filled hour and indicator
    imputation[1, 6:8] = False  # supervised target and its indicator
    imputation[2, 1:3] = False  # income and its indicator
    cases = {
        "prediction": _CoordinateMask("prediction", "prediction", np.argwhere(natural & ~prediction)),
        "imputation_30": _CoordinateMask("imputation_30_r0", "imputation_30", np.argwhere(natural & ~imputation)),
    }
    monkeypatch.setattr(tasks, "_load_cases", lambda name, data_root: (data, cases))
    return data


def test_prediction_preserves_prepared_fills_and_returns_entire_target_group(grouped):
    result = tgb.load_downstream_task("grouped", "prediction")
    assert set(result) == {"observed", "targets"}
    pd.testing.assert_frame_equal(result["observed"], grouped.test.iloc[:, :6])
    pd.testing.assert_frame_equal(result["targets"], grouped.test.iloc[:, 6:])
    assert result["observed"].loc[0, "income"] == 10.0
    assert result["targets"].loc[2, "label"] == "(missing)"
    assert result["targets"].loc[2, "label__missing"] == "missing"


def test_imputation_moves_fills_and_indicators_with_their_feature_group(grouped):
    result = tgb.load_downstream_task("grouped", "imputation_30")
    observed, targets = result["observed"], result["targets"]
    expected_hidden = pd.DataFrame(False, index=grouped.test.index,
                                   columns=grouped.test.columns)
    expected_hidden.loc[0, ["stamp__days", "stamp__hours", "stamp__missing"]] = True
    expected_hidden.loc[1, ["label", "label__missing"]] = True
    expected_hidden.loc[2, ["income", "income__missing"]] = True
    pd.testing.assert_frame_equal(observed.isna(), expected_hidden)
    pd.testing.assert_frame_equal(targets.notna(), expected_hidden)
    pd.testing.assert_frame_equal(observed.combine_first(targets), grouped.test)
    assert targets.loc[0, "stamp__hours"] == 6.0  # prepared fill is a target value
    assert targets.loc[0, "stamp__missing"] == "present"
    assert observed.loc[0, "income"] == 10.0  # unselected fill stays visible
    assert observed.loc[0, "income__missing"] == "missing"
    assert targets.loc[1, "label"] == "no"
    assert targets.loc[1, "label__missing"] == "present"
    assert observed.loc[2, "label"] == "(missing)"


def test_task_names_and_unavailable_conditions(grouped):
    assert tgb.list_downstream_tasks("grouped") == ("prediction", "imputation_30")
    with pytest.raises(ValueError, match="not applicable"):
        tgb.load_downstream_task("grouped", "imputation_50")


def test_unknown_task_fails_without_loading_data(monkeypatch):
    monkeypatch.setattr(tasks, "_load_cases", lambda *a: pytest.fail("must not load data"))
    with pytest.raises(ValueError, match="Unknown downstream task"):
        tgb.load_downstream_task("anything", "invalid")


def test_returned_frames_do_not_mutate_the_prepared_dataset(grouped):
    for name in ("prediction", "imputation_30"):
        result = tgb.load_downstream_task("grouped", name)
        result["observed"].iloc[0, 0] = -1.0
        result["targets"].iloc[0, 0] = "changed" if name == "prediction" else -2.0
    assert grouped.test.loc[0, "age"] == 20.0
    assert grouped.test.loc[0, "label"] == "yes"


def test_public_loader_has_no_mask_configuration_options():
    assert tuple(inspect.signature(tgb.load_downstream_task).parameters) == ("name", "task", "data_root")


@pytest.fixture
def consumer(bench, tmp_path):
    from tabgen_bench.downstream import creator_tool

    data = creator_tool.load_mask_creation_data(NAME, bench.root, max_rows=None)
    source = bench.root / "downstream" / f"{NAME}.npz"
    creator_tool.load_or_create_masks(data, source)
    root = tmp_path / "consumer"
    prepared = root / "prepared" / NAME
    prepared.mkdir(parents=True)
    shutil.copyfile(bench.root / "prepared" / NAME / "test.parquet", prepared / "test.parquet")
    (root / "downstream").mkdir()
    shutil.copyfile(source, root / "downstream" / source.name)
    shutil.copyfile(source.with_suffix(".json"), root / "downstream" / source.with_suffix(".json").name)
    return root


def test_fixed_tasks_use_only_prepared_test_and_existing_masks(consumer, monkeypatch):
    from tabgen_bench.downstream import creator_tool
    from tabgen_bench import preprocessing, validation

    def forbidden(*a, **kw):
        pytest.fail("consumer must not read raw/train data, create masks or recover eligibility")

    monkeypatch.setenv("TABGEN_BENCH_DATA", str(consumer))
    for name in ("load_mask_creation_data", "build_mask_cases", "load_or_create_masks", "_observed"):
        monkeypatch.setattr(creator_tool, name, forbidden)
    monkeypatch.setattr(preprocessing, "load_raw_splits", forbidden)
    monkeypatch.setattr(validation, "verify_prepared_dataframes", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    available = tgb.list_downstream_tasks(NAME)
    assert available[0] == "prediction" and "imputation_30" in available
    path = consumer / "downstream" / f"{NAME}.npz"
    original = path.read_bytes(), path.with_suffix(".json").read_bytes()
    prepared = pd.read_parquet(consumer / "prepared" / NAME / "test.parquet")
    # The synthetic benchmark's actual target comes from its shipped record.
    from tabgen_bench.registry import read_source
    target = read_source(NAME)["dataset"]["target"]
    for name in available:
        result = tgb.load_downstream_task(NAME, name)
        assert len(result["observed"]) == len(result["targets"]) == len(prepared) == 500
        assert result["observed"].index.equals(result["targets"].index)
        if name == "prediction":
            assert target in result["targets"] and target not in result["observed"]
        else:
            assert target in result["observed"] and target in result["targets"]
            assert result["targets"][target].notna().any()
            assert result["observed"][target].notna().any()
            assert result["observed"].columns.equals(result["targets"].columns)
            pd.testing.assert_frame_equal(result["observed"].combine_first(result["targets"]),
                                           prepared.loc[:, result["observed"].columns])
    assert not (consumer / "raw").exists()
    assert not (consumer / "prepared" / NAME / "train.parquet").exists()
    assert not (consumer / "prepared" / NAME / "validation.parquet").exists()
    assert original == (path.read_bytes(), path.with_suffix(".json").read_bytes())


def test_missing_mask_artifacts_use_fixed_repository(consumer, monkeypatch):
    monkeypatch.setenv("TABGEN_BENCH_MASK_REPO", "ignored/repo")
    monkeypatch.setenv("TABGEN_BENCH_MASK_REVISION", "ignored-revision")
    path = consumer / "downstream" / f"{NAME}.npz"
    path.unlink()
    path.with_suffix(".json").unlink()

    def download(**kwargs):
        assert kwargs["repo_id"] == "Granica/TabGenBench"
        assert kwargs["revision"] == "main"
        assert set(kwargs["allow_patterns"]) == {f"downstream/{NAME}.npz", f"downstream/{NAME}.json"}
        raise FileNotFoundError("Remote mask artifacts are not published")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.raises(FileNotFoundError, match="not published"):
        tgb.load_downstream_task(NAME, "prediction", data_root=consumer)
    assert not path.exists() and not path.with_suffix(".json").exists()


@pytest.mark.parametrize("missing_suffixes", [(".npz", ".json"), (".npz",), (".json",)])
def test_missing_masks_download_once_and_preserve_existing_files(consumer, monkeypatch, missing_suffixes):
    path = consumer / "downstream" / f"{NAME}.npz"
    artifacts = {suffix: path.with_suffix(suffix).read_bytes() for suffix in (".npz", ".json")}
    expected = tgb.load_downstream_task(NAME, "imputation_50", data_root=consumer)
    for suffix in missing_suffixes:
        path.with_suffix(suffix).unlink()
    calls = []

    def download(**kwargs):
        from pathlib import Path

        calls.append(kwargs)
        assert kwargs["repo_id"] == "Granica/TabGenBench"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "main"
        assert set(kwargs["allow_patterns"]) == {f"downstream/{NAME}{s}" for s in missing_suffixes}
        for suffix in missing_suffixes:
            (Path(kwargs["local_dir"]) / "downstream" / f"{NAME}{suffix}").write_bytes(artifacts[suffix])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    result = tgb.load_downstream_task(NAME, "imputation_50", data_root=consumer)
    for key in expected:
        pd.testing.assert_frame_equal(result[key], expected[key])
    assert "prediction" in tgb.list_downstream_tasks(NAME, data_root=consumer)
    assert len(calls) == 1
    assert all(path.with_suffix(s).read_bytes() == content for s, content in artifacts.items())


def test_invalid_download_is_not_installed(consumer, monkeypatch):
    from pathlib import Path

    path = consumer / "downstream" / f"{NAME}.npz"
    original = path.read_bytes()
    manifest = path.with_suffix(".json").read_text()
    path.unlink()

    def download(**kwargs):
        (Path(kwargs["local_dir"]) / "downstream" / f"{NAME}.npz").write_bytes(original + b"corrupt")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.raises(ValueError, match="NPZ checksum"):
        tgb.load_downstream_task(NAME, "prediction", data_root=consumer)
    assert not path.exists()
    assert path.with_suffix(".json").read_text() == manifest
    assert not list(consumer.glob(".mask-download-*"))


def test_modified_prepared_test_is_rejected(consumer):
    path = consumer / "prepared" / NAME / "test.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "num_a"] += 1.0
    frame.to_parquet(path, index=False)
    with pytest.raises(tgb.PreparedMismatchError, match="content checksum"):
        tgb.load_downstream_task(NAME, "prediction", data_root=consumer)


@pytest.mark.parametrize("fault", ["schema", "content", "rows", "settings", "bounds", "duplicate",
                                   "coordinates", "missing_binding", "file", "keys", "protocol"])
def test_consumer_rejects_incompatible_or_corrupt_mask_artifacts(consumer, fault):
    from tabgen_bench.downstream.data import _array_digest, _json_digest

    path = consumer / "downstream" / f"{NAME}.npz"
    sidecar = path.with_suffix(".json")
    manifest = json.loads(sidecar.read_text())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if fault == "schema":
        manifest["prepared_checksums"]["schema"] = "wrong"
    elif fault == "content":
        manifest["prepared_checksums"]["test"] = "wrong"
    elif fault == "rows":
        manifest["identity"]["test_row_ids_sha256"] = "wrong"
    elif fault == "settings":
        manifest["identity"]["settings"]["repeats"] = 3
    elif fault == "bounds":
        arrays["positions_1"][0, 0] = 500
    elif fault == "duplicate":
        arrays["positions_1"][1] = arrays["positions_1"][0]
    elif fault == "coordinates":
        manifest["cases"][1]["encoded_sha256"] = "wrong"
    elif fault == "missing_binding":
        del manifest["prepared_checksums"]
    elif fault == "protocol":
        manifest["identity"]["version"] = "downstream-masks-v1"
    elif fault == "keys":
        arrays["extra"] = np.zeros((0, 2), dtype=np.uint8)
    if fault in ("bounds", "duplicate"):
        manifest["cases"][1]["encoded_sha256"] = _array_digest(arrays["positions_1"])
    manifest["identity_sha256"] = _json_digest(manifest["identity"])
    np.savez_compressed(path, **arrays)
    import hashlib
    manifest["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if fault == "file":
        path.write_bytes(path.read_bytes() + b"corrupt")
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        tgb.list_downstream_tasks(NAME, data_root=consumer)
