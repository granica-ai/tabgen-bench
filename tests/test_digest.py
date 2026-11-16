"""``compute_dataframe_digest``: equal for equal content however it is held, different otherwise."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from tabgen_bench.validation import compute_dataframe_digest


def frame():
    return pd.DataFrame({"x": [1.5, 2.0, -0.0, 3.25], "c": ["a", "bb", "", "ünï"],
                         "y": [1.0, 2.0, 3.0, 4.0]})


def test_digest_is_stable_for_this_frame():
    # a pinned value: if this changes, DIGEST_LABEL must be bumped and every
    # shipped tgb_prepared.json re-recorded
    assert compute_dataframe_digest(frame()) == (
        "905141c836ffc399aa0dac8cc82af4e64e52aa709b3bee018bc1897b6a2f1348")


def test_digest_ignores_how_the_content_is_stored():
    base = compute_dataframe_digest(frame())
    arrow = frame()
    arrow["c"] = arrow["c"].astype("string[pyarrow]")
    assert compute_dataframe_digest(arrow) == base
    for compression in ("snappy", "zstd", None):
        buffer = io.BytesIO()
        frame().to_parquet(buffer, compression=compression)
        assert compute_dataframe_digest(pd.read_parquet(io.BytesIO(buffer.getvalue()))) == base
    assert compute_dataframe_digest(frame().astype({"y": "float32"}).astype({"y": "float64"})) == base


@pytest.mark.parametrize("change", [
    lambda f: f.assign(x=[1.5, 2.0, 0.0, 3.25]),                 # sign of zero
    lambda f: f.assign(c=["a", "bc", "", "ünï"]),                # one cell
    lambda f: f.assign(c=["ab", "b", "", "ünï"]),                # a boundary moved
    lambda f: f[["c", "x", "y"]],                                # column order
    lambda f: f.iloc[::-1].reset_index(drop=True),               # row order
    lambda f: f.rename(columns={"y": "z"}),                      # a name
    lambda f: f.iloc[:3],                                        # fewer rows
    lambda f: f.assign(x=np.nextafter(f["x"].to_numpy(), np.inf)),  # one ulp
    lambda f: f.assign(y=f["y"].astype(int).astype(str)),        # number vs its text
])
def test_digest_changes_with_any_change_of_content(change):
    assert compute_dataframe_digest(change(frame())) != compute_dataframe_digest(frame())


def test_digest_ignores_how_a_text_column_is_chunked_or_sliced():
    import pyarrow as pa

    words = [f"w{i}" * (i % 4) for i in range(50)]          # empty strings included
    numbers = np.arange(50, dtype="float64")
    base = compute_dataframe_digest(pd.DataFrame({"n": numbers, "c": words}))
    for cuts in ([0, 50], [0, 1, 50], [0, 7, 7, 31, 50], [0, 49, 50]):
        chunked = pa.chunked_array([pa.array(words[a:b], pa.large_string())
                                    for a, b in zip(cuts, cuts[1:])])
        column = pd.Series(pd.arrays.ArrowExtensionArray(chunked))
        assert compute_dataframe_digest(pd.DataFrame({"n": numbers, "c": column})) == base, cuts
    # a slice of a longer array: its buffers start before its first value
    padded = pa.array(["zz", "pad"] + words + ["tail"], pa.large_string()).slice(2, 50)
    column = pd.Series(pd.arrays.ArrowExtensionArray(pa.chunked_array([padded])))
    assert compute_dataframe_digest(pd.DataFrame({"n": numbers, "c": column})) == base


def test_digest_survives_a_parquet_file_with_many_row_groups(tmp_path):
    big = pd.DataFrame({"n": np.arange(5000, dtype="float64"),
                        "c": [f"label {i % 97}" for i in range(5000)]})
    path = tmp_path / "big.parquet"
    big.to_parquet(path, row_group_size=300, index=False)
    assert compute_dataframe_digest(pd.read_parquet(path)) == compute_dataframe_digest(big)
    assert compute_dataframe_digest(pd.read_parquet(path, dtype_backend="pyarrow")) == compute_dataframe_digest(big)


def test_every_nan_is_the_same_nan():
    plain = pd.DataFrame({"x": np.array([np.nan, 1.0])})
    negative = pd.DataFrame({"x": np.array([-np.nan, 1.0])})
    assert compute_dataframe_digest(plain) == compute_dataframe_digest(negative)


def test_a_null_in_a_text_column_is_refused():
    with pytest.raises(ValueError, match="nulls"):
        compute_dataframe_digest(pd.DataFrame({"c": ["a", None]}))
