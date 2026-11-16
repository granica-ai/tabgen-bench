"""Direct prediction and imputation cases over checked TabGenBench data.

Use ``list_downstream_tasks`` and ``load_downstream_task`` for prepared test
inputs and targets. Mask creation and full validation live in ``creator_tool``;
maintenance helpers remain available here through lazy compatibility exports.
"""

from .tasks import list_downstream_tasks, load_downstream_task

__all__ = [
    "list_downstream_tasks", "load_downstream_task",
    "DownstreamDataset", "load_mask_creation_data",
    "build_mask_cases", "load_masks", "load_or_create_masks", "validate_mask_case",
]


def __getattr__(name: str):
    if name in ("DownstreamDataset", "load_mask_creation_data", "build_mask_cases", "load_masks",
                "load_or_create_masks", "validate_mask_case"):
        from . import creator_tool

        return getattr(creator_tool, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
