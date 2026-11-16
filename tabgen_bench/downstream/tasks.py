"""Prepared test inputs and targets for the fixed downstream task protocol.

Feature groups move together, including indicator columns and prepared fill
values. This interface partitions data; it does not select scoring cells or
compute metrics. Training data is loaded separately with ``load_table``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..registry import default_data_root
from .data import _load_prepared_task_data
from .masks import _ensure_fixed_masks, _load_fixed_masks

_TASKS = ("prediction", "imputation_30", "imputation_50")


def _available_tasks(cases):
    return tuple(task for task in _TASKS if task in cases)


def _load_cases(name, data_root):
    root = Path(data_root if data_root is not None else default_data_root())
    data = _load_prepared_task_data(name, root)
    _ensure_fixed_masks(data, root)
    cases = _load_fixed_masks(data, root / "downstream" / f"{name}.npz")
    return data, cases


def list_downstream_tasks(name: str, *, data_root: str | Path | None = None) -> tuple[str, ...]:
    """Applicable fixed-protocol tasks for a locally prepared benchmark table.

    Reads the prepared test split and full-test mask artifacts, downloading
    missing masks to ``<data_root>/downstream/<name>.npz``. Equivalent masking rates on
    narrow tables are omitted, as are conditions without eligible cells.
    """
    _, cases = _load_cases(name, data_root)
    return _available_tasks(cases)


def load_downstream_task(
    name: str,
    task: str,
    *,
    data_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Return ``observed`` and ``targets`` from the entire prepared test split.

    Task names come from :func:`list_downstream_tasks`. Uses seed 42 and exactly one mask
    pass, with no sampling or protocol options. Requires prepared test data and
    fixed mask artifacts, downloading them from the configured Hub repository
    when missing. Never reads raw data or generates masks.

    Prediction returns prepared feature groups as inputs and the supervised
    target's whole group as targets. Imputation samples every group, including
    the supervised target group. Each selected group, including its indicators,
    moves from inputs to targets for that row. Unselected target cells are null.

    Prepared fills and labels are preserved, including fills in target groups.
    Only deliberately hidden inputs become null. These are data partitions,
    not scoring eligibility masks; indicators and originally missing fills are
    not filtered from the returned targets. Both frames keep prepared test-row
    order and prepared column order. No training frame or scoring result is
    returned. Returned frames are independent copies.
    """
    if task not in _TASKS:
        raise ValueError(f"Unknown downstream task {task!r}; expected one of {_TASKS}")
    data, cases = _load_cases(name, data_root)
    if task not in cases:
        available = _available_tasks(cases)
        raise ValueError(f"{name}: task {task!r} is not applicable; available tasks: {available}")
    # Keep each feature and its indicators together when separating inputs and targets.
    if task == "prediction":
        target_group = next(group for group in data.feature_groups if data.target in group)
        features = [column for column in data.columns if column not in target_group]
        target_columns = [column for column in data.columns if column in target_group]
        return {"observed": data.test.loc[:, features].copy(),
                "targets": data.test.loc[:, target_columns].copy()}

    columns = list(data.columns)
    case = cases[task]
    positions = np.zeros(data.test.shape, dtype=bool)
    positions[case.positions[:, 0], case.positions[:, 1]] = True
    hidden = pd.DataFrame(False, index=data.test.index, columns=columns)
    for group in data.feature_groups:
        indices = [data.columns.index(column) for column in group]
        selected = positions[:, indices].any(axis=1)
        # A group may include prepared fills that were originally missing.
        # Move the whole prepared group, rather than only its observed members.
        hidden.loc[selected, list(group)] = True
    return {"observed": data.test.mask(hidden, other=np.nan),
            "targets": data.test.where(hidden, other=np.nan)}
