from qdrant_client import models

from manufacturing_ai_copilot.rag import hybrid_index_builder


class FakeClient:
    def __init__(self, *, exists: bool):
        self.exists = exists
        self.checked_collection = None
        self.deleted_collection = None
        self.create_kwargs = None

    def collection_exists(self, collection_name):
        self.checked_collection = collection_name
        return self.exists

    def delete_collection(self, *, collection_name):
        self.deleted_collection = collection_name

    def create_collection(self, **kwargs):
        self.create_kwargs = kwargs


def test_recreate_hybrid_collection_deletes_existing_collection(monkeypatch):
    fake_client = FakeClient(exists=True)
    received_url = None

    def fake_get_qdrant_client(url):
        nonlocal received_url
        received_url = url
        return fake_client

    monkeypatch.setattr(
        hybrid_index_builder,
        "get_qdrant_client",
        fake_get_qdrant_client,
    )

    result = hybrid_index_builder.recreate_hybrid_collection()

    assert result is fake_client
    assert received_url == hybrid_index_builder.QDRANT_HYBRID_URL
    assert (
        fake_client.checked_collection
        == hybrid_index_builder.QDRANT_HYBRID_COLLECTION_NAME
    )
    assert (
        fake_client.deleted_collection
        == hybrid_index_builder.QDRANT_HYBRID_COLLECTION_NAME
    )
    assert (
        fake_client.create_kwargs["collection_name"]
        == hybrid_index_builder.QDRANT_HYBRID_COLLECTION_NAME
    )
    dense_config = fake_client.create_kwargs["vectors_config"][
        hybrid_index_builder.DENSE_VECTOR_NAME
    ]

    sparse_config = fake_client.create_kwargs["sparse_vectors_config"][
        hybrid_index_builder.SPARSE_VECTOR_NAME
    ]

    assert dense_config.size == hybrid_index_builder.QDRANT_VECTOR_SIZE
    assert dense_config.distance == models.Distance(
        hybrid_index_builder.QDRANT_DISTANCE
    )

    assert sparse_config.modifier == models.Modifier.IDF


def test_recreate_hybrid_collection_skips_delete_when_missing(monkeypatch):
    fake_client = FakeClient(exists=False)

    def fake_get_qdrant_client(url):
        return fake_client

    monkeypatch.setattr(
        hybrid_index_builder,
        "get_qdrant_client",
        fake_get_qdrant_client,
    )

    result = hybrid_index_builder.recreate_hybrid_collection()

    assert result is fake_client
    assert (
        fake_client.checked_collection
        == hybrid_index_builder.QDRANT_HYBRID_COLLECTION_NAME
    )

    assert fake_client.deleted_collection is None
    assert (
        fake_client.create_kwargs["collection_name"]
        == hybrid_index_builder.QDRANT_HYBRID_COLLECTION_NAME
    )
