"""Download the benchmark's tables from Hugging Face.

The tables come from the dataset repository ``TabArena/BeyondArena``.
:func:`download` fetches a table's directory into ``<data root>/raw/`` at the
commit its origin record pins (:mod:`tabgen_bench.registry`), so a later
change to the repository cannot alter the benchmark; a name that is not a
table of the benchmark is skipped with a message.  :func:`download_tables` is the whole step a
user runs: download what ``raw/`` lacks, then verify every table against its
record (:mod:`tabgen_bench.validation`), which writes ``verified.json``.

    python -m tabgen_bench.download                       # every benchmark table, 4.8 GB
    python -m tabgen_bench.download --datasets sdss_17 kick
"""

from __future__ import annotations

from pathlib import Path

from .registry import (
    DATA_ROOT,
    HF_REPO,
    benchmark_tables,
    default_data_root,
    read_source,
)
from .validation import ReproducibilityError, verification_line, verify_source
from .datasets import dataset_dir


def download(names, data_root: Path = DATA_ROOT, repo: str = HF_REPO) -> Path:
    """Fetch the named dataset directories into ``raw/``, skipping what is present.

    Each table is fetched at the commit its origin record pins, so a later
    change to the repository cannot alter the benchmark.  A name without a
    record is not a table of this benchmark: it is skipped, with a printed
    message, and nothing is fetched for it.
    ``snapshot_download`` resumes partial files and re-downloads nothing that
    matches the requested revision, so this is safe to call again.

    Args:
        names: Table names to fetch, e.g. ``["sdss_17", "kick"]``.  Each is a
            top-level directory of the repository; every file under it is
            fetched except the text-embedding caches (``*text_cache*``),
            which the benchmark does not read.  An empty list fetches
            nothing.
        data_root: The data root.  Files are written to ``<data_root>/raw/``,
            which is created if it does not exist, one directory per table.
            Defaults to ``$TABGEN_BENCH_DATA``, or ``~/.cache/tabgen_bench``
            when that is unset.
        repo: The Hugging Face dataset repository to download from.  The
            benchmark's records pin commits of ``TabArena/BeyondArena``, the
            default, so another repository only makes sense for a mirror
            that holds the same commits.

    Returns:
        The ``raw`` directory, ``<data_root>/raw``.  A table's files are under
        ``raw/<name>/``, and the client's own bookkeeping (which commit each
        file came from) under ``raw/.cache/huggingface/``.

    Raises:
        huggingface_hub.errors.HfHubHTTPError: The repository, the pinned
            commit or a file could not be reached.  A name that is not a
            benchmark table is not an error: it is skipped and reported.
    """
    from huggingface_hub import snapshot_download

    raw = Path(data_root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    by_revision: dict[str, list[str]] = {}
    for name in names:
        source = read_source(name)
        if source is None:
            print(f"{name}: not a table of this benchmark, skipped "
                  "(tabgen_bench.list_tables() lists them)", flush=True)
            continue
        by_revision.setdefault(source["origin"]["revision"], []).append(name)
    for revision, group in by_revision.items():
        snapshot_download(
            repo_id=repo, repo_type="dataset", local_dir=str(raw), revision=revision,
            allow_patterns=[f"{name}/**" for name in group],
            ignore_patterns=["*text_cache*"],
        )
    return raw


def raw_present(name: str, data_root: Path) -> bool:
    try:
        dataset_dir(name, Path(data_root) / "raw")
    except (FileNotFoundError, ValueError):
        return False
    return True


def download_tables(tables=None, *, data_root: Path | None = None) -> dict[str, dict | None]:
    """Verify ``tables`` against their records, downloading what fails; cut nothing.

    The whole step a user runs to get raw tables they can trust.  The names
    are checked first: one that is not a table of the benchmark is reported
    and skipped before anything is fetched.  Each table is then verified
    against its record (:func:`~tabgen_bench.validation.verify_source`):
    each recorded file present, the TGB raw schema, and the size and
    sha256 of every file.  A table that fails -- a file missing, as after an
    interrupted download, or a file that does not match its record -- is
    downloaded again (:func:`download`) and verified once more: the files
    that failed are removed first, so the client fetches them rather than
    trusting what is on disk.  A table that passes gets a ``verified.json``,
    and one status line is printed per table.  No split is cut and no
    prepared table is written.

    A table whose files are untouched since an earlier ``verified.json`` is
    not hashed again: its stored report is returned with ``reused`` True
    (see :func:`~tabgen_bench.validation.verify_source`).

    Args:
        tables: Table names to fetch and verify, e.g. ``["sdss_17", "kick"]``.
            None or an empty list means every table of the benchmark
            (:func:`~tabgen_bench.registry.benchmark_tables`).  A name that
            is not a benchmark table is skipped by :func:`download` and not
            verified.
        data_root: The data root, keyword-only.  Raw files are read from and
            written to ``<data_root>/raw/``, and the evidence of each check
            to ``<data_root>/table_sources/<name>/verified.json``.  None, the
            default, means ``$TABGEN_BENCH_DATA`` as it is set when the
            function is called, or ``~/.cache/tabgen_bench`` when it is
            unset.

    Returns:
        ``{name: report}``, one entry per name in the order given.  A report
        is the dictionary written to ``verified.json``: the repository and
        commit checked against (``repo_id``, ``revision``), when
        (``verified_at``), the schema matched, ``files_checked`` and each
        file's measured size and sha256 (``files``), and whether the download
        cache agrees with the pinned commit (``download_cache_agrees``).  The
        report is None for a name that is not a benchmark table.

    Raises:
        FileNotFoundError: A file the record lists is still missing from
            ``raw/`` after the table was downloaded again.
        tabgen_bench.validation.SchemaMismatchError: A raw table does not
            have the TGB raw schema, even freshly downloaded.
        tabgen_bench.validation.ChecksumMismatchError: A file's size or
            sha256 differs from its record, even freshly downloaded.
        huggingface_hub.errors.HfHubHTTPError: A table could not be
            downloaded.

        The first such failure stops the call: that table's earlier
        ``verified.json`` is removed, later tables are not checked, and
        nothing is returned.
    """
    data_root = Path(data_root or default_data_root())
    names = list(tables) if tables else list(benchmark_tables())
    reports = {}
    for name in names:
        source = read_source(name)
        if source is None:
            print(f"{name}: not a table of this benchmark, skipped "
                  "(tabgen_bench.list_tables() lists them)", flush=True)
            reports[name] = None
            continue
        try:
            reports[name] = verify_source(name, data_root)
        except (FileNotFoundError, ReproducibilityError) as problem:
            print(f"{problem}; downloading the table again", flush=True)   # names the table
            remove_recorded_files(name, data_root, source)
            download([name], data_root)
            reports[name] = verify_source(name, data_root, recheck=True)
        print(verification_line(name, reports[name], data_root), flush=True)
    return reports


def remove_recorded_files(name: str, data_root: Path, source: dict) -> list[Path]:
    """Delete ``name``'s recorded files from ``raw/`` so a download fetches them anew.

    Only the paths ``files.json`` lists are removed, with the download
    client's own note of each (``raw/.cache/huggingface/download/<path>.metadata``),
    which is what would otherwise tell the client the file is already the
    pinned one.  Nothing else under ``raw/`` is touched.  Returns the paths
    removed.
    """
    raw = Path(data_root) / "raw"
    cache = raw / ".cache" / "huggingface" / "download"
    removed = []
    for entry in source["files"]:
        for path in (raw / entry["path"], cache / (entry["path"] + ".metadata")):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="tables to download and verify (default: the "
                             "benchmark's tables, registry.TABLES)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    download_tables(args.datasets, data_root=args.data_root)


if __name__ == "__main__":
    main()
