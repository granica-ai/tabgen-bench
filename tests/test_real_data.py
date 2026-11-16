"""The chain on the real tables.  Needs the downloaded data.

Set ``TABGEN_BENCH_DATA`` to a data root whose ``raw/`` holds the tables; the
tests are skipped for a table that is not there.  Each one builds the table
as TGB in a fresh root and holds it to the shipped digests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tabgen_bench as tgb
from tabgen_bench import creator_tool, download, preprocessing, registry

pytestmark = pytest.mark.data
# small tables, one that BeyondArena treats as IID, one temporal and one grouped
TABLES = ("sdss_17", "kick", "5g_energy_consumption")


def data_root() -> Path | None:
    configured = os.environ.get(registry.DATA_ENV)
    return Path(configured) if configured else None


@pytest.mark.parametrize("name", TABLES)
def test_tgb_build_matches_the_shipped_digests(name, tmp_path):
    source = data_root()
    if source is None or not download.raw_present(name, source):
        pytest.skip(f"{name} is not downloaded under ${registry.DATA_ENV}")
    os.symlink(source / "raw", tmp_path / "raw")
    spec = preprocessing.prepare_table(name, tmp_path)
    assert spec["tgb"] is True
    assert spec["reproducibility"] == {
        "raw_schema_verified": True, "raw_checksum_verified": True,
        "split_checksum_verified": True, "column_spec_verified": True,
        "matches_tgb": True}
    assert spec["digests"] == registry.read_source(name)["prepared"]["digests"]
    splits = tgb.load_table(name, data_root=tmp_path)
    assert {s: len(f) for s, f in splits.items()} == registry.read_source(name)["split"]["rows"]


@pytest.mark.data
def test_the_creator_tool_reproduces_the_dropped_columns_list():
    source = data_root()
    if source is None or not all(download.raw_present(n, source) for n in registry.TABLES):
        pytest.skip(f"the 28 tables are not all downloaded under ${registry.DATA_ENV}")
    assert creator_tool.find_dropped_columns(registry.TABLES, source) == registry.DROPPED_COLUMNS
