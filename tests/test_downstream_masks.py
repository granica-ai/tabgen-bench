"""Task eligibility, nonleaking source groups and reusable cache integrity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tabgen_bench.downstream.data import _array_digest, _json_digest
from tabgen_bench.downstream.creator_tool import DownstreamDataset
from tabgen_bench.downstream.creator_tool import (
    STORAGE_VERSION,
    build_mask_cases, load_masks, load_or_create_masks,
)


def dataset(rows=60):
    frame = pd.DataFrame(
        {
            "a": np.arange(rows, dtype=float),
            "stamp__days": np.arange(rows, dtype=float),
            "stamp__hours": np.arange(rows, dtype=float) % 24,
            "stamp__missing": ["present"] * rows,
            "color": ["red", "blue"] * (rows // 2),
            "label": ["yes", "no"] * (rows // 2),
        }
    )
    observed = np.ones(frame.shape, dtype=bool)
    observed[0, 0] = False
    observed[1, 1:3] = False
    observed[2, 4] = False
    observed[3, 5] = False
    return DownstreamDataset(
        "toy",
        frame,
        frame,
        {
            c: "categorical" if c in ("stamp__missing", "color", "label") else "numerical"
            for c in frame
        },
        "label",
        observed,
        observed,
        np.arange(rows),
        "test-fingerprint",
        metadata={"indicator_columns": ["stamp__missing"]},
        feature_groups=(
            ("a",),
            ("stamp__days", "stamp__hours", "stamp__missing"),
            ("color",),
            ("label",),
        ),
    )


def test_prediction_hides_target_and_imputation_samples_it_without_scoring_indicators():
    data = dataset()
    cases = build_mask_cases(data, seed=5)
    assert cases[0].family == "prediction"
    assert not cases[0].observed[:, 5].any()
    np.testing.assert_array_equal(cases[0].score[:, 5], data.observed_test[:, 5])
    assert not cases[0].score[:, :5].any()
    for case in cases[1:]:
        assert case.observed[:, 5].any() and case.score[:, 5].any()
        assert not case.score[:, 3].any()
        assert not (case.score & ~data.observed_test).any()
        assert not (case.observed & case.score).any()
        # Derived day and hour columns of the same timestamp disappear together.
        np.testing.assert_array_equal(case.observed[:, 1], case.observed[:, 2])
        hidden_stamp = data.observed_test[:, 1] & ~case.observed[:, 1]
        assert not case.observed[hidden_stamp, 3].any()
        # Four source groups, including the supervised target:
        # every task must preserve an observed source feature somewhere.
        assert case.observed[:, [0, 1, 2, 4, 5]].any(axis=1).all()


def test_no_singleton_condition_is_built():
    cases = build_mask_cases(dataset(), include_blocks=True)
    assert [case.family for case in cases] == [
        "prediction", "imputation_30", "imputation_50", "schema_block"]


def test_masks_reproduce_and_change_with_seed():
    data = dataset()
    first = build_mask_cases(data, seed=2)
    again = build_mask_cases(data, seed=2)
    changed = build_mask_cases(data, seed=3)
    assert [c.name for c in first] == [c.name for c in again]
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a.observed, b.observed)
        np.testing.assert_array_equal(a.score, b.score)
    assert not np.array_equal(first[1].score, changed[1].score)


def test_imputation_samples_target_derivatives_and_indicators_together():
    data = replace(dataset(), target="stamp__days")
    for case in build_mask_cases(data)[1:]:
        selected = case.score[:, 1]
        visible = data.observed_test[:, 1] & ~selected
        assert selected.any() and visible.any()
        assert not case.observed[np.ix_(selected, [1, 2, 3])].any()
        np.testing.assert_array_equal(case.observed[np.ix_(visible, [1, 2, 3])],
                                      data.observed_test[np.ix_(visible, [1, 2, 3])])
        assert not case.score[:, 3].any()


def test_equivalent_low_dimensional_tasks_are_not_counted_repeatedly():
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    valid = np.ones(frame.shape, dtype=bool)
    data = DownstreamDataset(
        "one-feature",
        frame,
        frame,
        {"x": "numerical", "y": "numerical"},
        "y",
        valid,
        valid,
        np.arange(2),
        "identity",
    )
    cases = build_mask_cases(data, include_blocks=True)
    assert [c.family for c in cases if c.family != "schema_block"] == ["prediction", "imputation_30"]


def test_equivalent_rates_are_omitted_even_when_draws_would_differ():
    rows = 80
    frame = pd.DataFrame({"a": np.arange(rows), "b": np.arange(rows), "y": np.arange(rows)})
    valid = np.ones(frame.shape, dtype=bool)
    data = DownstreamDataset(
        "two-features",
        frame,
        frame,
        dict.fromkeys(frame.columns, "numerical"),
        "y",
        valid,
        valid,
        np.arange(rows),
        "identity",
    )
    cases = build_mask_cases(data, rates=(0.3, 0.4))
    assert [case.family for case in cases] == ["prediction", "imputation_30"]


def test_minimal_clinical_sepsis_keeps_30_percent_without_changing_existing_masks(tmp_path):
    data = replace(dataset(), name="sepsis_survival_minimal_clinical_records")
    previous = build_mask_cases(data, rates=(0.5,))
    path = tmp_path / "sepsis.npz"
    cases = load_or_create_masks(data, path)
    assert [case.family for case in cases] == ["prediction", "imputation_30", "imputation_50"]
    for old, new in zip(previous, (cases[0], cases[2]), strict=True):
        np.testing.assert_array_equal(old.observed, new.observed)
        np.testing.assert_array_equal(old.score, new.score)
    hidden = data.observed_test & ~cases[1].observed
    hidden_groups = np.column_stack([
        hidden[:, [data.columns.index(column) for column in group]].any(axis=1)
        for group in data.feature_groups
    ])
    assert (hidden_groups.sum(axis=1) == 1).all()
    loaded = load_masks(data, path)
    assert [case.family for case in loaded] == [case.family for case in cases]
    for expected, actual in zip(cases, loaded, strict=True):
        np.testing.assert_array_equal(expected.observed, actual.observed)
        np.testing.assert_array_equal(expected.score, actual.score)


def test_cache_reloads_saved_npz_and_manifest_and_rejects_identity_change(tmp_path):
    data = dataset()
    path = tmp_path / "masks.npz"
    written = load_or_create_masks(data, path)
    loaded = load_or_create_masks(data, path)
    assert path.is_file() and path.with_suffix(".json").is_file()
    for a, b in zip(written, loaded, strict=True):
        np.testing.assert_array_equal(a.score, b.score)
    with pytest.raises(ValueError, match="identity"):
        load_or_create_masks(data, path, seed=7)
    with pytest.raises(ValueError, match="identity"):
        load_or_create_masks(replace(data, fingerprint="other"), path)
    changed_observed = data.observed_test.copy()
    changed_observed[10, 0] = False
    with pytest.raises(ValueError, match="identity"):
        load_or_create_masks(replace(data, observed_test=changed_observed), path)


def test_cache_checks_both_file_and_individual_array_hashes(tmp_path):
    data = dataset()
    path = tmp_path / "masks.npz"
    load_or_create_masks(data, path)
    original = path.read_bytes()
    path.write_bytes(original + b"corrupt")
    with pytest.raises(ValueError, match="NPZ checksum"):
        load_or_create_masks(data, path)
    path.write_bytes(original)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["positions_1"][0, 1] ^= 1
    np.savez_compressed(path, **arrays)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="array checksum"):
        load_or_create_masks(data, path)


def test_cache_refuses_partial_cache(tmp_path):
    data = dataset()
    path = tmp_path / "masks.npz"
    path.write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="Incomplete"):
        load_or_create_masks(data, path)


def test_cache_never_unpickles_object_arrays(tmp_path):
    data = dataset()
    path = tmp_path / "masks.npz"
    load_or_create_masks(data, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["positions_1"] = arrays["positions_1"].astype(object)
    np.savez_compressed(path, **arrays)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_or_create_masks(data, path)


@pytest.mark.parametrize("target", ["label", "stamp__days"])
def test_compact_cases_reconstruct_groups_missingness_and_target_derivatives(tmp_path, target):
    data = replace(dataset(), target=target)
    path = tmp_path / "compact.npz"
    expected = load_or_create_masks(data, path, include_blocks=True)
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["storage_version"] == STORAGE_VERSION
    assert "test_row_ids" not in manifest["identity"]
    assert manifest["identity"]["test_row_ids_sha256"] == _array_digest(data.test_row_ids)
    assert all(info["encoding"] == "coordinates" for info in manifest["cases"])
    with np.load(path, allow_pickle=False) as archive:
        assert all(name.startswith("positions_") for name in archive.files)
        for i, case in enumerate(expected):
            np.testing.assert_array_equal(archive[f"positions_{i}"],
                                          np.argwhere(data.observed_test & ~case.observed))
        assert sum(archive[name].nbytes for name in archive.files) < sum(
            c.observed.nbytes + c.score.nbytes for c in expected)
    actual = load_masks(data, path, include_blocks=True)
    for a, b in zip(expected, actual, strict=True):
        assert (a.name, a.family, a.repeat) == (b.name, b.family, b.repeat)
        np.testing.assert_array_equal(a.observed, b.observed)
        np.testing.assert_array_equal(a.score, b.score)


def test_compact_masks_support_wide_tables_and_rows_without_feature_truth(tmp_path):
    frame = pd.DataFrame(np.zeros((6, 261)), columns=[f"x{i}" for i in range(261)])
    observed = np.ones(frame.shape, dtype=bool)
    observed[0, :-1] = False
    data = DownstreamDataset("wide", frame, frame, dict.fromkeys(frame, "numerical"),
                             "x260", observed, observed, np.arange(6), "wide-identity")
    path = tmp_path / "wide.npz"
    expected = load_or_create_masks(data, path)
    families = ["prediction", "imputation_30", "imputation_50"]
    names = ["prediction", *[f"{family}_r0" for family in families[1:]]]
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert [case["family"] for case in manifest["cases"]] == families
    assert [case["name"] for case in manifest["cases"]] == names
    with np.load(path, allow_pickle=False) as archive:
        assert archive["positions_1"].dtype == np.uint16
        np.testing.assert_array_equal(archive["positions_1"][archive["positions_1"][:, 0] == 0],
                                      [[0, 260]])
    loaded = load_masks(data, path)
    assert [case.family for case in loaded] == families
    assert [case.name for case in loaded] == names
    for a, b in zip(expected, loaded, strict=True):
        np.testing.assert_array_equal(a.observed, b.observed)
        np.testing.assert_array_equal(a.score, b.score)


def test_target_only_cache_records_prediction_and_imputation_positions(tmp_path):
    frame = pd.DataFrame({"y": [1.0, 2.0]})
    observed = np.ones(frame.shape, dtype=bool)
    data = DownstreamDataset("target-only", frame, frame, {"y": "numerical"}, "y",
                             observed, observed, np.arange(2), "target-identity")
    path = tmp_path / "prediction.npz"
    load_or_create_masks(data, path)
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files == ["positions_0", "positions_1"]
        np.testing.assert_array_equal(archive["positions_0"], [[0, 0], [1, 0]])
        np.testing.assert_array_equal(archive["positions_1"], [[0, 0], [1, 0]])
    assert len(load_masks(data, path)) == 2


def test_coordinate_rows_above_uint16_range_roundtrip(tmp_path):
    rows = 65537
    frame = pd.DataFrame({"x": np.zeros(rows), "y": np.ones(rows)})
    observed = np.ones(frame.shape, dtype=bool)
    data = DownstreamDataset("many-rows", frame.iloc[:1], frame,
                             {"x": "numerical", "y": "numerical"}, "y",
                             observed[:1], observed, np.arange(rows), "many-rows-identity")
    path = tmp_path / "many.npz"
    load_or_create_masks(data, path, rates=(0.5,))
    with np.load(path, allow_pickle=False) as archive:
        assert archive["positions_0"].dtype == np.uint32
        np.testing.assert_array_equal(archive["positions_0"][-1], [65536, 1])
    loaded = load_masks(data, path, rates=(0.5,))
    assert loaded[0].score[:, 1].all()
    assert loaded[1].score.sum(axis=1).tolist() == [1] * rows
    assert loaded[1].score[:, 0].any() and loaded[1].score[:, 1].any()


def test_empty_coordinate_matrix_roundtrip(tmp_path):
    frame = pd.DataFrame({"y": [1.0, 2.0]})
    observed = np.zeros(frame.shape, dtype=bool)
    data = DownstreamDataset("unknown-target", frame, frame, {"y": "numerical"}, "y",
                             observed, observed, np.arange(2), "unknown-target-identity")
    path = tmp_path / "empty.npz"
    load_or_create_masks(data, path)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["positions_0"].shape == (0, 2)
    assert not load_masks(data, path)[0].observed.any()


@pytest.mark.parametrize("fault", ["bounds", "duplicate", "order", "shape", "dtype",
                                   "dense_hash", "rows", "storage", "keys", "missing",
                                   "partial_group", "encoding", "repeat"])
def test_invalid_coordinate_artifacts_rejected_even_with_updated_checksums(tmp_path, fault):
    data = dataset()
    path = tmp_path / "bad.npz"
    load_or_create_masks(data, path)
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if fault == "bounds":
        arrays["positions_1"][0, 0] = len(data.test)
    elif fault == "duplicate":
        arrays["positions_1"][1] = arrays["positions_1"][0]
    elif fault == "order":
        arrays["positions_1"] = arrays["positions_1"][::-1]
    elif fault == "shape":
        arrays["positions_1"] = arrays["positions_1"].ravel()
    elif fault == "dtype":
        arrays["positions_1"] = arrays["positions_1"].astype(np.float64)
    elif fault == "dense_hash":
        manifest["cases"][1]["observed_sha256"] = "wrong"
    elif fault == "rows":
        data = replace(data, test_row_ids=data.test_row_ids[::-1])
    elif fault == "storage":
        manifest["storage_version"] = "unknown"
    elif fault == "keys":
        arrays["extra"] = np.zeros(1, dtype=np.uint8)
    elif fault == "missing":
        arrays["positions_1"][0] = [0, 0]  # naturally missing in this fixture
    elif fault == "partial_group":
        positions = arrays["positions_1"]
        stamp = np.flatnonzero(positions[:, 1] == 1)
        assert len(stamp)
        arrays["positions_1"] = np.delete(positions, stamp[0], axis=0)
    elif fault == "repeat":
        manifest["cases"][1]["repeat"] = 1
    elif fault == "encoding":
        manifest["cases"][1]["encoding"] = "unknown"
    for i, info in enumerate(manifest["cases"]):
        info["encoded_sha256"] = _array_digest(arrays[f"positions_{i}"])
    np.savez_compressed(path, **arrays)
    manifest["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_masks(data, path)


@pytest.mark.parametrize("storage", [None, "downstream-group-masks-v1", "unknown"])
@pytest.mark.parametrize("loader", [load_masks, load_or_create_masks])
def test_unsupported_storage_is_rejected_without_rewriting(tmp_path, storage, loader):
    data = dataset()
    path = tmp_path / "unsupported.npz"
    load_or_create_masks(data, path)
    sidecar = path.with_suffix(".json")
    manifest = json.loads(sidecar.read_text())
    if storage is None:
        manifest.pop("storage_version")
    else:
        manifest["storage_version"] = storage
    sidecar.write_text(json.dumps(manifest))
    original = path.read_bytes(), sidecar.read_bytes()
    with pytest.raises(ValueError, match="Unsupported mask storage version"):
        loader(data, path)
    assert original == (path.read_bytes(), sidecar.read_bytes())


@pytest.mark.parametrize("function", [build_mask_cases, load_masks, load_or_create_masks])
def test_repetition_count_is_not_an_api_option(function, tmp_path):
    args = (dataset(),) if function is build_mask_cases else (dataset(), tmp_path / "masks.npz")
    with pytest.raises(TypeError, match="repeats"):
        function(*args, repeats=2)


def test_each_mask_family_has_exactly_one_pass():
    cases = build_mask_cases(dataset(), include_blocks=True)
    assert all(case.repeat == 0 for case in cases)
    assert len({case.family for case in cases}) == len(cases)


def test_multiple_pass_cache_is_rejected_without_rewriting(tmp_path):
    data = dataset()
    path = tmp_path / "multiple.npz"
    load_or_create_masks(data, path)
    sidecar = path.with_suffix(".json")
    manifest = json.loads(sidecar.read_text())
    manifest["identity"]["settings"]["repeats"] = 3
    manifest["identity_sha256"] = _json_digest(manifest["identity"])
    sidecar.write_text(json.dumps(manifest))
    original = path.read_bytes(), sidecar.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        load_or_create_masks(data, path)
    assert original == (path.read_bytes(), sidecar.read_bytes())


def test_creator_omits_repeat_and_score_hash_from_new_manifests(tmp_path):
    path = tmp_path / "clean.npz"
    original = load_or_create_masks(dataset(), path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert all("repeat" not in case and "score_sha256" not in case
               for case in manifest["cases"])
    loaded = load_masks(dataset(), path)
    for a, b in zip(original, loaded, strict=True):
        np.testing.assert_array_equal(a.observed, b.observed)
        np.testing.assert_array_equal(a.score, b.score)
