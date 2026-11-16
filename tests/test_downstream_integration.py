"""Direct-task cases use the benchmark's real verification chain and creator helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

import tabgen_bench as tgb
from tabgen_bench.downstream import creator_tool as downstream_creator

from conftest import NAME


def test_checked_data_and_masks_roundtrip(bench, tmp_path, monkeypatch):
    monkeypatch.setenv("TABGEN_BENCH_DATA", str(bench.root))
    data = downstream_creator.load_mask_creation_data(NAME, max_rows=32)
    assert len(data.test) == 32 and len(data.train) == 4000
    _, positions = tgb.load_split_indices(NAME)
    assert set(data.test_row_ids) <= set(positions["test"])
    path = tmp_path / "cases"
    written = downstream_creator.load_or_create_masks(data, path)
    # Frozen artifacts load without asking the generator for new random draws.
    monkeypatch.setattr(np.random, "default_rng",
                        lambda *a, **kw: pytest.fail("must reuse stored masks"))
    loaded = downstream_creator.load_masks(data, path)
    assert [case.name for case in loaded] == [case.name for case in written]
    for a, b in zip(written, loaded, strict=True):
        downstream_creator.validate_mask_case(data, b)
        np.testing.assert_array_equal(a.observed, b.observed)
        np.testing.assert_array_equal(a.score, b.score)


def test_load_masks_does_not_create_missing_artifacts(bench, tmp_path):
    data = downstream_creator.load_mask_creation_data(NAME, bench.root)
    assert len(data.test) == 500
    path = tmp_path / "missing" / "masks.npz"
    with pytest.raises(FileNotFoundError, match="Mask cache does not exist"):
        downstream_creator.load_masks(data, path)
    assert not path.parent.exists()



def test_full_test_masks_are_created_and_verified_without_modifying_files(bench, tmp_path):
    data = downstream_creator.load_mask_creation_data(NAME, bench.root)
    assert len(data.test) == 500
    path = tmp_path / f"{NAME}.npz"
    created = downstream_creator.load_or_create_masks(data, path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["identity"]["test_rows"] == 500
    original = path.read_bytes(), path.with_suffix(".json").read_bytes()
    loaded = downstream_creator.load_masks(data, path)
    assert [case.name for case in loaded] == [case.name for case in created]
    assert original == (path.read_bytes(), path.with_suffix(".json").read_bytes())
    with pytest.raises(ValueError, match="identity"):
        downstream_creator.load_masks(data, path, seed=7)
