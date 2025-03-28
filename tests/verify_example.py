from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from minimalkv import get_store_from_url

from plateau.io.eager import read_table, store_dataframes_as_dataset


@pytest.fixture
def store_url():
    dataset_dir = TemporaryDirectory()
    store_url = f"hfs://{dataset_dir.name}"

    yield get_store_from_url(store_url)

    dataset_dir.cleanup()


def test_example_fast_read(store_url):
    # con = duckdb.connect()
    # con.execute("CREATE TABLE my_df (a INTEGER, b VARCHAR)")
    # con.execute("INSERT INTO my_df VALUES (1, 'a'), (2, 'b')")
    df = pd.DataFrame(
        {
            "A": 1.0,
            "B": [
                pd.Timestamp("20130102"),
                pd.Timestamp("20130102"),
                pd.Timestamp("20130103"),
                pd.Timestamp("20130103"),
            ],
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    )

    store_dataframes_as_dataset(
        store_url, "partitioned_dataset", [df], partition_on="B"
    )
    read_table(
        dataset_uuid="partitioned_dataset",
        store=store_url,
    )

