"""Shared mask-case representation and artifact format versions.

Eligibility, generation and full cache validation live in ``creator_tool``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .data import _array_digest, _json_digest

MASK_VERSION = "masks-v1"
STORAGE_VERSION = "downstream-coordinates-v1"
MASK_REPO = "Granica/TabGenBench"
# Replace with the upload commit SHA when the mask release is published.
MASK_REVISION = "main"


def _ensure_fixed_masks(data, root):
    """Download missing masks from the benchmark's Hub repository."""
    root = Path(root)
    relative = Path("downstream") / f"{data.name}.npz"
    paths = (relative, relative.with_suffix(".json"))
    missing = [path for path in paths if not (root / path).is_file()]
    if not missing:
        return
    from huggingface_hub import snapshot_download

    # Stage and validate the complete pair before installing missing files.
    # Existing artifacts are preserved, including a surviving half of a pair.
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mask-download-", dir=root) as temporary:
        staging = Path(temporary)
        (staging / relative.parent).mkdir(parents=True)
        for path in paths:
            if path not in missing:
                shutil.copyfile(root / path, staging / path)
        snapshot_download(
            repo_id=MASK_REPO, repo_type="dataset", revision=MASK_REVISION,
            allow_patterns=[path.as_posix() for path in missing],
            local_dir=str(staging),
        )
        _load_fixed_masks(data, staging / relative)
        (root / relative.parent).mkdir(parents=True, exist_ok=True)
        for path in missing:
            os.replace(staging / path, root / path)


def _coordinate_dtype(shape):
    return np.dtype(np.min_scalar_type(max(0, *[size - 1 for size in shape]))).newbyteorder("<")


@dataclass(frozen=True)
class _CoordinateMask:
    name: str
    family: str
    positions: np.ndarray


def _load_fixed_masks(data, path):
    """Read existing coordinates using prepared identity and artifact checksums.

    This reader never recovers missingness, generates masks or imports creator
    code. Full raw-dependent validation belongs to ``creator_tool``.
    """
    path = Path(path)
    # Coordinates live in the NPZ; the neighboring JSON describes and binds them
    # to the prepared data and fixed task protocol.
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(
            f"Fixed mask NPZ and JSON artifacts are required: {path}; "
            "create them with tabgen_bench.downstream.creator_tool first")
    try:
        manifest = json.loads(sidecar.read_text())
        if manifest.get("storage_version") != STORAGE_VERSION:
            raise ValueError("Fixed tasks require coordinate mask artifacts; regenerate with creator_tool")
        # Verify the masks were created for these exact test values and schema.
        if "prepared_checksums" not in manifest:
            raise ValueError("Mask artifact lacks prepared checksums; regenerate with creator_tool")
        if manifest["prepared_checksums"] != data.checksums:
            raise ValueError("Mask artifact prepared-content or schema checksum mismatch")
        identity = manifest["identity"]
        # Check the recorded identity's integrity before comparing its fields.
        if manifest["identity_sha256"] != _json_digest(identity):
            raise ValueError("Mask artifact identity checksum mismatch")
        # Consumers use the full test split, seed 42 and one pass. Extra creator
        # identity fields can remain, but every consumer-required field must match.
        expected = {
            "version": MASK_VERSION, "dataset": data.name, "columns": list(data.columns),
            "kinds": {c["name"]: c["kind"] for c in data.schema["columns"]},
            "target": data.target, "feature_groups": [list(g) for g in data.feature_groups],
            "indicator_columns": [c["name"] for c in data.schema["columns"] if "indicator_of" in c],
            "test_rows": len(data.test), "test_row_ids_sha256": data.row_ids_checksum,
            "settings": {"seed": 42, "repeats": 1, "rates": [0.3, 0.5], "include_blocks": False},
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError("Fixed mask identity differs from prepared schema, rows or task settings")
        # Bind the manifest to the complete NPZ before decoding its arrays.
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["npz_sha256"]:
            raise ValueError("Mask artifact NPZ checksum mismatch")
        cases, names, families = {}, set(), set()
        shape = data.test.shape
        with np.load(path, allow_pickle=False) as arrays:
            # Require one coordinate array per case, with no missing or extra arrays.
            if set(arrays.files) != {f"positions_{i}" for i in range(len(manifest["cases"]))}:
                raise ValueError("Mask artifact array keys differ from its manifest")
            for index, info in enumerate(manifest["cases"]):
                name, family = info["name"], info["family"]
                # Each supported task family may appear once, with a unique name.
                if (not isinstance(name, str) or not name or name in names or family in families
                        or family not in ("prediction", "imputation_30", "imputation_50")
                        or info["encoding"] != "coordinates"):
                    raise ValueError("Invalid or duplicate fixed mask case")
                positions = arrays[f"positions_{index}"]
                # Each pair is a zero-based [test-row, prepared-column] position.
                # The dtype must match the unsigned integer width chosen by the writer.
                if (positions.ndim != 2 or positions.shape[1] != 2
                        or positions.dtype != _coordinate_dtype(shape)):
                    raise ValueError("Mask positions must be a two-column unsigned integer matrix")
                if _array_digest(positions) != info["encoded_sha256"]:
                    raise ValueError("Mask coordinate checksum mismatch")
                rows, columns = positions[:, 0], positions[:, 1]
                if (rows >= shape[0]).any() or (columns >= shape[1]).any():
                    raise ValueError("Mask coordinates are out of bounds")
                # Strict row-major order also rules out duplicate coordinates;
                # empty and single-position arrays satisfy this check naturally.
                ordered = ((rows[1:] > rows[:-1])
                           | ((rows[1:] == rows[:-1]) & (columns[1:] > columns[:-1])))
                if not ordered.all():
                    raise ValueError("Mask coordinates must be unique and in row-major order")
                # Keep validated coordinates read-only and index cases by task name.
                positions.flags.writeable = False
                cases[family] = _CoordinateMask(name, family, positions)
                names.add(name)
                families.add(family)
        # Imputation cases can be omitted when inapplicable; prediction is required.
        if "prediction" not in cases:
            raise ValueError("Mask artifact is missing the prediction task")
        return cases
    except (OSError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"Invalid fixed mask artifact: {error}") from error


def __getattr__(name: str):
    # Preserve the previous public import paths without duplicating implementation.
    if name in ("build_mask_cases", "load_masks", "load_or_create_masks", "validate_mask_case"):
        from . import creator_tool

        return getattr(creator_tool, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
