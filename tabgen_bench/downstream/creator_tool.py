"""Creator-side downstream data checks, eligibility, masks and cache validation.

These functions use raw observations to create and fully validate mask
artifacts. Creator-side data and mask structures live here; prepared test-data
loading lives in ``data`` and stored-coordinate loading in ``masks``. Prepared
input/target views are exposed by ``tasks``. No model or metric is run here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .data import _array_digest, _json_digest
from .masks import MASK_VERSION, STORAGE_VERSION, _coordinate_dtype

DATA_VERSION = "downstream-data-v1"


@dataclass(frozen=True)
class DownstreamDataset:
    """Prepared values with aligned, true-observation masks in the same schema.

    ``feature_groups`` partitions related prepared columns, such as the day and
    hour of one raw timestamp. Indicators can occur in these groups but are
    never imputation targets. ``test_row_ids`` are positions in the raw table,
    rather than positions in the possibly subsampled test frame.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    kinds: Mapping[str, str]
    target: str
    observed_train: np.ndarray
    observed_test: np.ndarray
    test_row_ids: np.ndarray
    fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    feature_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        if not self.name or not self.fingerprint:
            raise ValueError("Dataset name and fingerprint must be nonempty")
        columns = tuple(self.train.columns)
        if len(set(columns)) != len(columns) or not all(isinstance(c, str) for c in columns):
            raise ValueError("Dataset columns must have unique string names")
        if tuple(self.test.columns) != columns or set(self.kinds) != set(columns):
            raise ValueError("Train, test and kinds must have the same column schema")
        if self.target not in columns:
            raise ValueError("Prediction target is not in the prepared schema")
        if any(k not in ("numerical", "categorical") for k in self.kinds.values()):
            raise ValueError("Column kinds must be numerical or categorical")
        indicators = tuple(self.metadata.get("indicator_columns", ()))
        if not set(indicators) <= set(columns) or self.target in indicators:
            raise ValueError("Invalid missing-indicator columns")
        for name, frame in (("observed_train", self.train), ("observed_test", self.test)):
            array = np.asarray(getattr(self, name))
            if array.dtype != np.bool_ or array.shape != frame.shape:
                raise ValueError(f"{name} must be a boolean array matching its frame")
            array = array.copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        row_ids = np.asarray(self.test_row_ids)
        if (
            row_ids.ndim != 1
            or len(row_ids) != len(self.test)
            or not np.issubdtype(row_ids.dtype, np.integer)
            or len(np.unique(row_ids)) != len(row_ids)
            or (row_ids < 0).any()
        ):
            raise ValueError("test_row_ids must contain one distinct nonnegative integer per row")
        row_ids = np.array(row_ids, dtype=np.int64, copy=True)
        row_ids.flags.writeable = False
        object.__setattr__(self, "test_row_ids", row_ids)
        groups = tuple(tuple(group) for group in self.feature_groups)
        if groups:
            flattened = [c for group in groups for c in group]
            if any(not group for group in groups) or len(flattened) != len(set(flattened)):
                raise ValueError("Feature groups must be nonempty and disjoint")
            if set(flattened) != set(columns):
                raise ValueError("Feature groups must partition the full prepared schema")
        else:
            groups = tuple((column,) for column in columns)
        object.__setattr__(self, "feature_groups", groups)
        object.__setattr__(self, "kinds", MappingProxyType(dict(self.kinds)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "train", self.train.copy().reset_index(drop=True))
        object.__setattr__(self, "test", self.test.copy().reset_index(drop=True))

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.train.columns)

    @property
    def indicator_columns(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("indicator_columns", ()))


@dataclass(frozen=True)
class MaskCase:
    """Visible and scored cells, both in the complete prepared column schema."""

    name: str
    family: str
    repeat: int
    observed: np.ndarray
    score: np.ndarray

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.family, str)
            or not self.family
            or isinstance(self.repeat, bool)
            or not isinstance(self.repeat, int)
            or self.repeat < 0
        ):
            raise ValueError("Mask case requires a name, family and nonnegative repeat")
        observed, score = np.asarray(self.observed), np.asarray(self.score)
        if (
            observed.dtype != np.bool_
            or score.dtype != np.bool_
            or observed.ndim != 2
            or observed.shape != score.shape
        ):
            raise ValueError("Observed and score masks must be matching two-dimensional booleans")
        if (observed & score).any():
            raise ValueError("Scored cells must be hidden")
        for name, array in (("observed", observed), ("score", score)):
            array = array.copy()
            array.flags.writeable = False
            object.__setattr__(self, name, array)


def _observed(raw: pd.DataFrame, prepared: pd.DataFrame, columns: list[dict]) -> np.ndarray:
    """Recover true validity from raw values, before the benchmark filled them."""
    from ..preprocessing import CONVERTERS, _numbers, raw_column_of

    observed = np.zeros(prepared.shape, dtype=bool)
    for index, column in enumerate(columns):
        name = column["name"]
        if "indicator_of" in column:
            # Whether a source value is absent is itself a known observation.
            observed[:, index] = True
            continue
        source = raw[raw_column_of(column)]
        if column["kind"] == "categorical":
            valid = source.notna().to_numpy(copy=True)
            valid &= prepared[name].notna().to_numpy()
            valid &= prepared[name].astype(str).to_numpy() != "(missing)"
        else:
            converter = CONVERTERS.get(column.get("source"), _numbers)
            valid = np.isfinite(converter(source))
            valid &= np.isfinite(prepared[name].to_numpy(dtype=float))
        observed[:, index] = valid
    return observed


def load_mask_creation_data(
    name: str, data_root: str | Path | None = None, *, max_rows: int | None = None, seed: int = 42
) -> DownstreamDataset:
    """Load checked, already prepared TGB data; never download or prepare it.

    ``data_root`` defaults to ``TABGEN_BENCH_DATA`` or the package cache.
    All training and test rows are retained by default. An explicit test cap
    samples rows independently of
    their outcomes and preserves raw positional IDs. Categories in metadata
    come exclusively from genuinely observed training values.
    """
    from ..datasets import read_spec, spec_file, split_file
    from ..preprocessing import load_raw_splits, raw_column_of
    from ..registry import SPLITS, benchmark_tables, default_data_root
    from ..validation import (
        load_column_spec,
        load_split_indices,
        verify_prepared_dataframes,
    )

    if name not in benchmark_tables():
        raise KeyError(f"Unknown TabGenBench table {name!r}")
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer or None")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    root = Path(data_root if data_root is not None else default_data_root())
    required = [spec_file(root, name), *(split_file(root, name, s) for s in SPLITS)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Local prepared benchmark files are required: " + ", ".join(missing)
        )
    verification = verify_prepared_dataframes(name, root)
    spec = read_spec(name, root)
    column_spec = load_column_spec(name)
    if (
        spec["columns"] != column_spec["columns"]
        or spec["column_order"] != column_spec["column_order"]
    ):
        raise ValueError(f"{name}: local prepared schema differs from the benchmark specification")
    loaded = load_raw_splits(name, root, verify=True)
    record, positions = load_split_indices(name)
    if spec["target"] != loaded["task"]["target_column_name"]:
        raise ValueError(f"{name}: local prediction target differs from raw task metadata")
    train = pd.read_parquet(split_file(root, name, "train"))
    test = pd.read_parquet(split_file(root, name, "test"))
    expected = tuple(column_spec["column_order"])
    if tuple(train.columns) != expected or tuple(test.columns) != expected:
        raise ValueError(f"{name}: prepared column order differs from the benchmark")
    if any(len(loaded["splits"][s]) != len(positions[s]) for s in SPLITS):
        raise ValueError(f"{name}: raw split rows do not align with stored split positions")
    if len(train) != len(positions["train"]) or len(test) != len(positions["test"]):
        raise ValueError(f"{name}: prepared rows do not align with stored split positions")
    columns = column_spec["columns"]
    observed_train = _observed(loaded["splits"]["train"], train, columns)
    observed_test = _observed(loaded["splits"]["test"], test, columns)
    chosen = np.arange(len(test), dtype=np.int64)
    if max_rows is not None and len(chosen) > max_rows:
        entropy = int(_json_digest([DATA_VERSION, name, seed])[:16], 16)
        chosen = np.sort(np.random.default_rng(entropy).choice(chosen, max_rows, replace=False))
    test_row_ids = np.asarray(positions["test"], dtype=np.int64)[chosen]
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(raw_column_of(column), []).append(column["name"])
    categories = {
        column["name"]: sorted(set(train.loc[observed_train[:, index], column["name"]].astype(str)))
        for index, column in enumerate(columns)
        if column["kind"] == "categorical"
    }
    identity = {
        "version": DATA_VERSION,
        "name": name,
        "digests": verification["digests"],
        "column_spec": column_spec,
        "split_sha256": record["sha256"],
        "test_row_ids": test_row_ids.tolist(),
        "observed_train": _array_digest(observed_train),
        "observed_test": _array_digest(observed_test[chosen]),
    }
    return DownstreamDataset(
        name=name,
        train=train,
        test=test.iloc[chosen],
        kinds={c["name"]: c["kind"] for c in columns},
        target=spec["target"],
        observed_train=observed_train,
        observed_test=observed_test[chosen],
        test_row_ids=test_row_ids,
        fingerprint=_json_digest(identity),
        metadata={
            "version": DATA_VERSION,
            "task_type": loaded["task"]["problem_type"],
            "categories": categories,
            "indicator_columns": [c["name"] for c in columns if "indicator_of" in c],
            "train_row_ids": np.asarray(positions["train"], dtype=np.int64).tolist(),
            "test_positions": chosen.tolist(),
            "source_columns": {c["name"]: raw_column_of(c) for c in columns},
            "split_sha256": record["sha256"],
            "prepared_digests": verification["digests"],
            "column_spec_sha256": _json_digest(column_spec),
            "max_rows": max_rows,
            "selection_seed": seed,
        },
        feature_groups=tuple(tuple(group) for group in groups.values()),
    )


def _settings(seed, rates, include_blocks):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rates = tuple(float(rate) for rate in rates)
    if any(not np.isfinite(rate) or not 0 < rate < 1 for rate in rates):
        raise ValueError("Mask rates must lie strictly between zero and one")
    if len(set(rates)) != len(rates):
        raise ValueError("Mask rates must be distinct")
    if not isinstance(include_blocks, bool):
        raise ValueError("include_blocks must be a boolean")
    return {
        "seed": seed,
        "repeats": 1,  # fixed protocol metadata, not a configurable setting
        "rates": list(rates),
        "include_blocks": include_blocks,
    }


def _groups(dataset):
    columns = dataset.columns
    target = columns.index(dataset.target)
    indicators = {columns.index(name) for name in dataset.indicator_columns}
    target_group = next(group for group in dataset.feature_groups if dataset.target in group)
    withheld = {columns.index(name) for name in target_group}
    feature_groups = [
        np.array([columns.index(name) for name in group], dtype=np.int64)
        for group in dataset.feature_groups
        if any(columns.index(c) not in indicators for c in group)
    ]
    eligible = dataset.observed_test.copy()
    eligible[:, list(indicators)] = False
    return target, withheld, feature_groups, eligible


def validate_mask_case(dataset: DownstreamDataset, case: MaskCase) -> None:
    """Check the single-pass protocol, shape, missingness and score eligibility."""
    if case.repeat != 0:
        raise ValueError("Only one mask pass is supported; case repeat must be zero")
    if case.observed.shape != dataset.test.shape:
        raise ValueError("Cached mask shape differs from the dataset")
    if (case.observed & ~dataset.observed_test).any():
        raise ValueError("A mask reveals naturally missing cells")
    if (case.score & ~dataset.observed_test).any():
        raise ValueError("A mask scores naturally missing cells")
    target, withheld, _, eligible = _groups(dataset)
    if case.family == "prediction":
        if case.observed[:, list(withheld)].any():
            raise ValueError("Prediction target and its derivatives must be hidden")
        expected = np.zeros_like(case.score)
        expected[:, target] = dataset.observed_test[:, target]
        if not np.array_equal(case.score, expected):
            raise ValueError("Prediction masks must score only observed target values")
    elif (case.score & ~eligible).any():
        raise ValueError("Imputation masks cannot score missing indicators")


def build_mask_cases(
    dataset: DownstreamDataset,
    *,
    seed: int = 42,
    rates: tuple[float, ...] = (0.3, 0.5),
    include_blocks: bool = False,
) -> tuple[MaskCase, ...]:
    """Build one pass of prediction and random-subset imputation tasks.

    Random rates apply to eligible *source groups*, keeping timestamp siblings
    together. At least one eligible feature group stays visible when there are
    two or more. Rates that select the same number of groups on every row as
    an earlier rate are omitted, except that the minimal clinical sepsis
    dataset always retains the requested 30% condition. Optional ``schema_block`` tasks hide a
    contiguous interval of schema source groups, without claiming correlated
    missingness. Identical imputation masks are emitted once as well.
    """
    settings = _settings(seed, rates, include_blocks)
    target, withheld, groups, eligible = _groups(dataset)
    base = dataset.observed_test.copy()
    base[:, list(withheld)] = False
    score = np.zeros_like(base)
    score[:, target] = dataset.observed_test[:, target]
    cases = [MaskCase("prediction", "prediction", 0, base, score)]
    # Imputation samples every source group, including the supervised target.
    base = dataset.observed_test.copy()
    available_groups = (np.column_stack([eligible[:, group].any(axis=1) for group in groups])
                        if groups else np.empty((len(dataset.test), 0), dtype=bool))
    available = [np.flatnonzero(row) for row in available_groups]
    count_plans = set()
    keep_sepsis_30 = dataset.name == "sepsis_survival_minimal_clinical_records"
    rate_plans = []
    for rate in settings["rates"]:
        counts = tuple(
            min(max(1, int(round(rate * len(candidates)))), max(1, len(candidates) - 1))
            if len(candidates)
            else 0
            for candidates in available
        )
        if counts not in count_plans or (keep_sepsis_30 and rate == 0.3):
            rate_plans.append((rate, counts))
            count_plans.add(counts)
    seen = set()

    def append(family, selected):
        visible = base.copy()
        scored = np.zeros_like(base)
        hidden_groups = np.zeros_like(available_groups)
        for row, picked in enumerate(selected):
            hidden_groups[row, picked] = True
        for index, group in enumerate(groups):
            cells = np.ix_(hidden_groups[:, index], group)
            visible[cells] = False
            scored[cells] = eligible[cells]
        if not scored.any():
            return
        identity = (_array_digest(visible), _array_digest(scored))
        if identity in seen and not (keep_sepsis_30 and family == "imputation_30"):
            return
        seen.add(identity)
        cases.append(MaskCase(f"{family}_r0", family, 0, visible, scored))

    entropy = int(_json_digest([MASK_VERSION, dataset.name, seed, 0])[:16], 16)
    rng = np.random.default_rng(entropy)
    # The removed singleton condition drew one balanced group per row from this
    # stream before the rates did. The draws are still made, and discarded, so
    # the published 30% and 50% masks keep their coordinates.
    counts = np.zeros(len(groups), dtype=np.int64)
    for row in rng.permutation(len(available)):
        candidates = available[row]
        if len(candidates):
            rarest = candidates[counts[candidates] == counts[candidates].min()]
            counts[int(rng.choice(rarest))] += 1
    for rate, counts in rate_plans:
        sampling_rng = rng
        if keep_sepsis_30 and rate == 0.3:
            # Give the added condition its own stream so existing 50% masks stay fixed.
            entropy = int(_json_digest([MASK_VERSION, dataset.name, seed, "imputation_30"])[:16], 16)
            sampling_rng = np.random.default_rng(entropy)
        selected = []
        for candidates, count in zip(available, counts, strict=True):
            selected.append(sampling_rng.choice(candidates, size=count, replace=False).tolist())
        append(f"imputation_{rate * 100:g}", selected)
    if include_blocks:
        selected = []
        for candidates in available:
            size = len(candidates)
            count = min(max(1, int(round(0.5 * size))), max(1, size - 1)) if size else 0
            start = int(rng.integers(size - count + 1)) if size else 0
            selected.append(candidates[start : start + count].tolist())
        append("schema_block", selected)
    for case in cases:
        validate_mask_case(dataset, case)
    return tuple(cases)


def _cache_identity(dataset, settings):
    identity = {
        "version": MASK_VERSION,
        "dataset": dataset.name,
        "dataset_fingerprint": dataset.fingerprint,
        "columns": list(dataset.columns),
        "kinds": dict(dataset.kinds),
        "target": dataset.target,
        "indicator_columns": list(dataset.indicator_columns),
        "feature_groups": [list(group) for group in dataset.feature_groups],
        "observed_test_sha256": _array_digest(dataset.observed_test),
        "settings": settings,
    }
    identity["test_rows"] = len(dataset.test)
    identity["test_row_ids_sha256"] = _array_digest(dataset.test_row_ids)
    return identity


def _prepared_checksums(dataset):
    return {
        "test": dataset.metadata.get("prepared_digests", {}).get("test"),
        "schema": dataset.metadata.get("column_spec_sha256"),
    }


def _encode_coordinates(dataset, case):
    # Row-major, zero-based positions in the selected test frame and prepared
    # column order. Naturally missing cells need no entry: the data records them.
    return np.argwhere(dataset.observed_test & ~case.observed).astype(
        _coordinate_dtype(dataset.test.shape))


def _decode_coordinates(dataset, info, positions):
    shape = dataset.test.shape
    if (info["encoding"] != "coordinates" or positions.ndim != 2
            or positions.shape[1] != 2 or positions.dtype != _coordinate_dtype(shape)):
        raise ValueError("Coordinates must be a two-column unsigned integer matrix")
    rows, columns = positions[:, 0], positions[:, 1]
    if (rows >= shape[0]).any() or (columns >= shape[1]).any():
        raise ValueError("Mask coordinates are out of bounds")
    ordered = ((rows[1:] > rows[:-1])
               | ((rows[1:] == rows[:-1]) & (columns[1:] > columns[:-1])))
    if not ordered.all():
        raise ValueError("Mask coordinates must be unique and in row-major order")
    if not dataset.observed_test[rows, columns].all():
        raise ValueError("Coordinates cannot mask originally missing cells")
    hidden = np.zeros(shape, dtype=bool)
    hidden[rows, columns] = True
    target, _, groups, eligible = _groups(dataset)
    for group in groups:
        if info["family"] == "prediction" and target in group:
            continue
        selected = hidden[:, group].any(axis=1)
        if (selected & ~eligible[:, group].any(axis=1)).any():
            raise ValueError("A selected group has no observed feature truth")
        expected = dataset.observed_test[:, group] & selected[:, None]
        if not np.array_equal(hidden[:, group], expected):
            raise ValueError("Coordinates must hide complete source-feature groups")
    observed = dataset.observed_test & ~hidden
    if info["family"] == "prediction":
        score = np.zeros(shape, dtype=bool)
        score[:, target] = dataset.observed_test[:, target]
    else:
        score = hidden & eligible
    return MaskCase(info["name"], info["family"], info.get("repeat", 0), observed, score)


def load_or_create_masks(
    dataset: DownstreamDataset,
    path: str | Path,
    *,
    seed: int = 42,
    rates: tuple[float, ...] = (0.3, 0.5),
    include_blocks: bool = False,
) -> tuple[MaskCase, ...]:
    """Reuse an NPZ cache only after identity, content and semantic validation.

    ``path`` is an ``.npz`` filename, or a directory containing ``masks.npz``.
    A neighboring JSON manifest records dataset/settings identity, case names,
    whole-file SHA256 and individual mask hashes. Caches store two-column
    masked-cell coordinate matrices. Incomplete, mismatched or corrupt caches
    raise; they are never silently replaced. No pickle is read.
    """
    settings = _settings(seed, rates, include_blocks)
    return _cached_masks(dataset, path, settings, create=True)


def load_masks(
    dataset: DownstreamDataset,
    path: str | Path,
    *,
    seed: int = 42,
    rates: tuple[float, ...] = (0.3, 0.5),
    include_blocks: bool = False,
) -> tuple[MaskCase, ...]:
    """Load and verify existing masks; never generate or replace an artifact.

    Dataset identity and settings must match the manifest exactly, as with
    :func:`load_or_create_masks`. Missing files raise ``FileNotFoundError``;
    incomplete, corrupt or incompatible caches raise ``ValueError``.
    """
    settings = _settings(seed, rates, include_blocks)
    return _cached_masks(dataset, path, settings, create=False)


def _cached_masks(dataset, path, settings, *, create):
    path = Path(path)
    if path.suffix != ".npz":
        path = path / "masks.npz"
    manifest_path = path.with_suffix(".json")
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise ValueError("Incomplete mask cache: both NPZ and JSON manifest are required")
        try:
            manifest = json.loads(manifest_path.read_text())
            storage = manifest.get("storage_version")
            if storage != STORAGE_VERSION:
                raise ValueError("Unsupported mask storage version")
            if manifest.get("prepared_checksums") != _prepared_checksums(dataset):
                raise ValueError("Mask cache prepared-content or schema checksum mismatch")
            identity = _cache_identity(dataset, settings)
            if manifest["identity"] != identity or manifest["identity_sha256"] != _json_digest(
                identity
            ):
                raise ValueError("Mask cache identity differs from the dataset or task settings")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["npz_sha256"]:
                raise ValueError("Mask cache NPZ checksum mismatch")
            cases = []
            with np.load(path, allow_pickle=False) as arrays:
                expected_keys = {f"positions_{i}" for i in range(len(manifest["cases"]))}
                if set(arrays.files) != expected_keys:
                    raise ValueError("Mask cache array keys differ from its manifest")
                for index, info in enumerate(manifest["cases"]):
                    positions = arrays[f"positions_{index}"]
                    if _array_digest(positions) != info["encoded_sha256"]:
                        raise ValueError("Mask cache array checksum mismatch")
                    case = _decode_coordinates(dataset, info, positions)
                    if (_array_digest(case.observed) != info["observed_sha256"]
                            or ("score_sha256" in info
                                and _array_digest(case.score) != info["score_sha256"])):
                        raise ValueError("Mask cache reconstructed array checksum mismatch")
                    validate_mask_case(dataset, case)
                    cases.append(case)
            if not cases or cases[0].family != "prediction":
                raise ValueError("Mask cache is missing its prediction task")
            if len({case.name for case in cases}) != len(cases):
                raise ValueError("Mask cache contains duplicate task names")
            return tuple(cases)
        except (OSError, KeyError, TypeError, json.JSONDecodeError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"Invalid mask cache: {error}") from error
    if not create:
        raise FileNotFoundError(f"Mask cache does not exist: {path}")
    cases = build_mask_cases(dataset, seed=settings["seed"], rates=settings["rates"],
                             include_blocks=settings["include_blocks"])
    identity = _cache_identity(dataset, settings)
    arrays, records = {}, []
    for index, case in enumerate(cases):
        encoded = _encode_coordinates(dataset, case)
        info = {
            "name": case.name, "family": case.family,
            "encoding": "coordinates",
            "encoded_sha256": _array_digest(encoded),
            "observed_sha256": _array_digest(case.observed),
        }
        arrays[f"positions_{index}"] = encoded
        records.append(info)
    manifest = {
        "storage_version": STORAGE_VERSION,
        "prepared_checksums": _prepared_checksums(dataset),
        "identity": identity,
        "identity_sha256": _json_digest(identity),
        "cases": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as output:
            temporary = Path(output.name)
            np.savez_compressed(output, **arrays)
        manifest["npz_sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, path)
        temporary = None
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".json", mode="w", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(manifest, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, manifest_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return cases
