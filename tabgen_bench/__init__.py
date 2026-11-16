"""TabGenBench: 28 BeyondArena tables prepared for tabular data generation.

The package downloads each table from Hugging Face, verifies it, cuts the
benchmark's train / validation / test splits and applies one column policy,
so that every user works on the same rows and the same values::

    import tabgen_bench as tgb

    tgb.list_tables()                      # the 28 table names
    tgb.table_info("kick")                 # task, target, shape, licence, source
    splits = tgb.load_table("kick")        # {"train": df, "validation": df, "test": df}
    spec = tgb.load_spec("kick")           # columns, kinds, target, task
    tgb.download_tables(["kick"])          # what load_table does first on its own:
    tgb.prepare_tables(["kick"])           # fetch at the pinned commit, verify, build

Data is written under ``tgb.default_data_root()``: the ``TABGEN_BENCH_DATA``
environment variable when set, ``~/.cache/tabgen_bench`` otherwise.  Every
function also takes ``data_root=``.

Reproducibility is a chain, and a TGB build checks every link::

    TGB raw schema  ->  raw-data checksum  ->  stored row indices
        ->  split-file checksum  ->  train / validation / test
        ->  stored column specification  ->  content digests

A link that does not hold raises a :class:`ReproducibilityError`.

The names below are the public interface.  The submodules
(:mod:`~tabgen_bench.registry`, :mod:`~tabgen_bench.download`,
:mod:`~tabgen_bench.validation`, :mod:`~tabgen_bench.preprocessing`,
:mod:`~tabgen_bench.datasets`) hold the machinery and the non-TGB
options; each is also a command, ``python -m tabgen_bench.<module>``, and
``tabgen-bench <command>`` runs them by name.
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # what the benchmark is
    "TABLES",
    "list_tables",
    "table_info",
    "default_data_root",
    # building it
    "download_tables",
    "verify_tables",
    "prepare_tables",
    # using it
    "load_table",
    "load_spec",
    "load_raw",
    "load_split_indices",
    "compute_dataframe_digest",
    # direct prediction and imputation cases
    "list_downstream_tasks",
    "load_downstream_task",
    "DownstreamDataset",
    # what can go wrong
    "ReproducibilityError",
    "SchemaMismatchError",
    "ChecksumMismatchError",
    "SplitIndicesError",
    "ColumnSpecError",
    "PreparedMismatchError",
]

# Where each public name lives.  They are resolved on first use rather than
# imported here, so ``import tabgen_bench`` stays light and
# ``python -m tabgen_bench.<module>`` runs a module that has not already been
# imported by its own package.
_HOME = {
    "TABLES": "registry", "list_tables": "registry", "table_info": "registry",
    "default_data_root": "registry",
    "download_tables": "download",
    "verify_tables": "validation", "load_split_indices": "validation",
    "compute_dataframe_digest": "validation",
    "ReproducibilityError": "validation", "SchemaMismatchError": "validation",
    "ChecksumMismatchError": "validation", "SplitIndicesError": "validation",
    "ColumnSpecError": "validation", "PreparedMismatchError": "validation",
    "prepare_tables": "preprocessing",
    "load_table": "datasets", "load_spec": "datasets", "load_raw": "datasets",
    "DownstreamDataset": "downstream",
    "list_downstream_tasks": "downstream", "load_downstream_task": "downstream",
}

if TYPE_CHECKING:  # for editors and type checkers; never run
    from .datasets import load_raw, load_spec, load_table
    from .download import download_tables
    from .downstream import DownstreamDataset, list_downstream_tasks, load_downstream_task
    from .preprocessing import prepare_tables
    from .registry import TABLES, default_data_root, list_tables, table_info
    from .validation import (
        ChecksumMismatchError,
        ColumnSpecError,
        PreparedMismatchError,
        ReproducibilityError,
        SchemaMismatchError,
        SplitIndicesError,
        compute_dataframe_digest,
        load_split_indices,
        verify_tables,
    )


def __getattr__(name: str):
    if name in _HOME:
        import importlib

        value = getattr(importlib.import_module(f"{__name__}.{_HOME[name]}"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
