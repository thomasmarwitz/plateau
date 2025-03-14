import random
from urllib.parse import quote

import minimalkv
import numpy as np
import pandas as pd
import simplejson
from packaging import version

from plateau.core.common_metadata import (
    _get_common_metadata_key,
    make_meta,
    store_schema_metadata,
)
from plateau.core.dataset import DatasetMetadata, create_partition_key, naming
from plateau.core.urlencode import quote_indices, unquote_indices


def test_create_partition_key():
    key = create_partition_key(
        "my-uuid", "testtable", [("index1", "value1"), ("index2", "value2")]
    )
    assert key == "my-uuid/testtable/index1=value1/index2=value2/data"


def test_index_quote_roundtrip():
    indices = [
        (1, b"Muenchen"),
        ("location", b"Muenchen"),
        ("location", "München"),
        ("product", "å\\ øß"),
    ]
    expected = [
        ("1", "Muenchen"),
        ("location", "Muenchen"),
        ("location", "München"),
        ("product", "å\\ øß"),
    ]
    assert expected == unquote_indices(quote_indices(indices))


def testunquote_indices():
    index_strings = [
        "{}={}".format(quote("location"), quote("München".encode())),
        "{}={}".format(quote("product"), quote("å\\ øß".encode())),
    ]
    indices = unquote_indices(index_strings)
    assert indices == [("location", "München"), ("product", "å\\ øß")]


def test_dynamic_partitions(store):
    """Do not specify partitions in metadata, but read them dynamically from
    store."""
    partition_suffix = "suffix"
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    partition0_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-0")],
        f"{partition_suffix}.parquet",
    )
    partition1_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-1")],
        f"{partition_suffix}.parquet",
    )
    metadata = {"dataset_metadata_version": 4, "dataset_uuid": dataset_uuid}
    expected_partitions = {
        f"location=L-0/{partition_suffix}": {"files": {"core": partition0_core}},
        f"location=L-1/{partition_suffix}": {"files": {"core": partition1_core}},
    }
    expected_indices = {
        "location": {
            "L-0": [f"location=L-0/{partition_suffix}"],
            "L-1": [f"location=L-1/{partition_suffix}"],
        }
    }

    # put two partitions for two tables each to store
    store.put(
        f"{dataset_uuid}{naming.METADATA_BASE_SUFFIX}.json",
        simplejson.dumps(metadata).encode("utf-8"),
    )
    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store_schema_metadata(
        make_meta(
            pd.DataFrame({"location": [f"L-0/{partition_suffix}"]}),
            origin="stored",
        ),
        dataset_uuid,
        store,
        "core",
    )

    # instantiate metadata to write table metadatad
    core_schema = make_meta(
        pd.DataFrame(
            {
                "column_0": pd.Series([1], dtype=int),
                "column_1": pd.Series([1], dtype=int),
                "location": pd.Series(["str"]),
            }
        ),
        origin="core",
    )
    store_schema_metadata(core_schema, dataset_uuid, store, "core")

    dmd = DatasetMetadata.load_from_store(dataset_uuid, store)
    dmd = dmd.load_partition_indices()

    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    assert dmd_dict["indices"] == expected_indices


def test_dynamic_partitions_multiple_indices(store):
    """Do not specify partitions in metadata, but read them dynamically from
    store."""
    suffix = "suffix"
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    partition0_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-0"), ("product", "P-0")],
        f"{suffix}.parquet",
    )
    partition1_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-1"), ("product", "P-0")],
        f"{suffix}.parquet",
    )
    metadata = {"dataset_metadata_version": 4, "dataset_uuid": dataset_uuid}
    expected_partitions = {
        f"location=L-0/product=P-0/{suffix}": {"files": {"core": partition0_core}},
        f"location=L-1/product=P-0/{suffix}": {"files": {"core": partition1_core}},
    }
    expected_indices = {
        "location": {
            "L-0": [f"location=L-0/product=P-0/{suffix}"],
            "L-1": [f"location=L-1/product=P-0/{suffix}"],
        },
        "product": {
            "P-0": [
                f"location=L-0/product=P-0/{suffix}",
                f"location=L-1/product=P-0/{suffix}",
            ]
        },
    }

    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store_schema_metadata(
        make_meta(pd.DataFrame({"location": ["L-0"], "product": ["P-0"]}), origin="1"),
        dataset_uuid,
        store,
        "core",
    )

    dmd = DatasetMetadata.load_from_dict(metadata, store)
    dmd = dmd.load_partition_indices()
    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    # Sorting may differ in the index list. This is ok for runtime
    # but does produce flaky tests thus sort them.
    sorted_result = {
        column: {label: sorted(x) for label, x in index.items()}
        for column, index in dmd_dict["indices"].items()
    }
    assert sorted_result == expected_indices


def test_dynamic_partitions_with_garbage(store):
    """In case there are unknown files, dataset and indices still load
    correctly."""
    dataset_uuid = "uuid+namespace-attribute12_underscored"
    partition_suffix = "suffix"
    partition0_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-0"), ("product", "P-0")],
        f"{partition_suffix}.parquet",
    )
    partition1_core = create_partition_key(
        dataset_uuid,
        "core",
        [("location", "L-1"), ("product", "P-0")],
        f"{partition_suffix}.parquet",
    )
    metadata = {"dataset_metadata_version": 4, "dataset_uuid": dataset_uuid}
    expected_partitions = {
        f"location=L-0/product=P-0/{partition_suffix}": {
            "files": {"core": partition0_core}
        },
        f"location=L-1/product=P-0/{partition_suffix}": {
            "files": {"core": partition1_core}
        },
    }
    expected_indices = {
        "location": {
            "L-0": [f"location=L-0/product=P-0/{partition_suffix}"],
            "L-1": [f"location=L-1/product=P-0/{partition_suffix}"],
        },
        "product": {
            "P-0": [
                f"location=L-0/product=P-0/{partition_suffix}",
                f"location=L-1/product=P-0/{partition_suffix}",
            ]
        },
    }

    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store_schema_metadata(
        make_meta(pd.DataFrame({"location": ["L-0"], "product": ["P-0"]}), origin="1"),
        dataset_uuid,
        store,
        "core",
    )

    # the following files are garbage and should not interfere with the indices and/or partitions
    for suffix in ["", ".json", ".msgpack", ".my_own_file_format"]:
        store.put(f"this_should_not_exist{suffix}", b"ignore me")
        store.put(f"{dataset_uuid}/this_should_not_exist{suffix}", b"ignore me")
        store.put(
            "{}/{}/this_should_not_exist{}".format(dataset_uuid, "core", suffix),
            b"ignore me",
        )
        store.put(
            "{}/{}/location=L-0/this_should_not_exist{}".format(
                dataset_uuid, "core", suffix
            ),
            b"ignore me",
        )

    dmd = DatasetMetadata.load_from_dict(metadata, store)
    dmd = dmd.load_partition_indices()
    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    # Sorting may differ in the index list. This is ok for runtime
    # but does produce flaky tests thus sort them.
    sorted_result = {
        column: {label: sorted(x) for label, x in index.items()}
        for column, index in dmd_dict["indices"].items()
    }
    assert sorted_result == expected_indices


def test_dynamic_partitions_quote(store, metadata_version):
    """Do not specify partitions in metadata, but read them dynamically from
    store."""
    dataset_uuid = "uuid-namespace-attribute12_underscored"
    partition0_core = create_partition_key(
        dataset_uuid, "core", [("location", "München")], "data.parquet"
    )
    partition1_core = create_partition_key(
        dataset_uuid, "core", [("location", "å\\ øß")], "data.parquet"
    )
    metadata = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": dataset_uuid,
    }
    expected_partitions = {
        "location=M%C3%BCnchen/data": {"files": {"core": partition0_core}},
        "location=%C3%A5%5C%20%C3%B8%C3%9F/data": {"files": {"core": partition1_core}},
    }
    expected_indices = {
        "location": {
            "München": ["location=M%C3%BCnchen/data"],
            "å\\ øß": ["location=%C3%A5%5C%20%C3%B8%C3%9F/data"],
        }
    }

    store.put(partition0_core, b"test")
    store.put(partition1_core, b"test")
    store_schema_metadata(
        make_meta(pd.DataFrame({"location": ["L-0"]}), origin="1"),
        dataset_uuid,
        store,
        "core",
    )

    dmd = DatasetMetadata.load_from_dict(metadata, store)
    dmd = dmd.load_partition_indices()

    dmd_dict = dmd.to_dict()
    assert dmd_dict["partitions"] == expected_partitions
    assert dmd_dict["indices"] == expected_indices


def test_dask_partitions(metadata_version, tmp_path):
    """Create partitions for one table with dask and check that it can be read
    with plateau."""
    import dask.dataframe

    dataset_uuid = "uuid+namespace-attribute12_underscored"
    (tmp_path / dataset_uuid).mkdir()
    table_path = tmp_path / dataset_uuid / "core"
    table_path.mkdir()
    store = minimalkv.get_store_from_url(f"hfs://{tmp_path}")

    locations = [f"L-{i}" for i in range(2)]
    df = pd.DataFrame()
    for location in locations:
        core = pd.DataFrame(
            data={
                "date": np.array(
                    ["2017-11-23", "2017-11-23", "2017-11-24", "2017-11-24"]
                ),
                "product": np.array(["P-0", "P-1", "P-0", "P-1"]),
                "location": location,
                "value": np.array(random.sample(range(1, 100), 4)),
            }
        )
        df = pd.concat([df, core])

    ddf = dask.dataframe.from_pandas(df, npartitions=1)
    dask_kwargs = {}
    if version.parse(dask.__version__) >= version.parse("2022.4.2"):
        dask_kwargs["write_metadata_file"] = True
    dask.dataframe.to_parquet(
        ddf, str(table_path), partition_on=["location"], **dask_kwargs
    )

    partition0 = f"{dataset_uuid}/core/location=L-0/part.0.parquet"
    partition1 = f"{dataset_uuid}/core/location=L-1/part.0.parquet"
    metadata = {
        "dataset_metadata_version": metadata_version,
        "dataset_uuid": dataset_uuid,
    }
    expected_partitions = {
        "partitions": {
            "location=L-0": {"files": {"core": partition0}},
            "location=L-1": {"files": {"core": partition1}},
        }
    }
    expected_tables = {"tables": {"core": ["date", "product", "value"]}}

    store.put(
        f"{dataset_uuid}.by-dataset-metadata.json",
        simplejson.dumps(metadata).encode(),
    )

    metadata.update(expected_partitions)
    metadata.update(expected_tables)
    dmd = DatasetMetadata.load_from_store(dataset_uuid, store)
    actual_partitions = dmd.to_dict()["partitions"]
    # we partition on location ID which has two values
    assert len(actual_partitions) == 2
    assert dmd.partition_keys == ["location"]


def test_overlap_keyspace(store, metadata_version):
    dataset_uuid1 = "uuid+namespace-attribute12_underscored"
    dataset_uuid2 = "uuid+namespace-attribute12_underscored_ext"
    table = "core"

    for dataset_uuid in (dataset_uuid1, dataset_uuid2):
        partition0 = "location=L-0"
        partition0_key = f"{dataset_uuid}/{table}/{partition0}/data.parquet"
        metadata = {
            "dataset_metadata_version": metadata_version,
            "dataset_uuid": dataset_uuid,
        }

        # put two partitions for two tables each to store
        store.put(
            f"{dataset_uuid}{naming.METADATA_BASE_SUFFIX}.json",
            simplejson.dumps(metadata).encode("utf-8"),
        )
        store.put(partition0_key, b"test")
        store_schema_metadata(
            make_meta(pd.DataFrame({"location": ["L-0"]}), origin="1"),
            dataset_uuid,
            store,
            "core",
        )

    for dataset_uuid in (dataset_uuid1, dataset_uuid2):
        partition0_label = "location=L-0/data"
        partition0_key = f"{dataset_uuid}/{table}/{partition0_label}.parquet"
        expected_partitions = {"location=L-0/data": {"files": {"core": partition0_key}}}
        expected_indices = {"location": {"L-0": ["location=L-0/data"]}}
        assert DatasetMetadata.storage_keys(dataset_uuid, store) == [
            f"{dataset_uuid}{naming.METADATA_BASE_SUFFIX}.json",
            _get_common_metadata_key(dataset_uuid, "core"),
            partition0_key,
        ]
        dmd = DatasetMetadata.load_from_store(dataset_uuid, store)
        dmd = dmd.load_partition_indices()
        dmd_dict = dmd.to_dict()
        assert dmd_dict["partitions"] == expected_partitions
        assert dmd_dict["indices"] == expected_indices
