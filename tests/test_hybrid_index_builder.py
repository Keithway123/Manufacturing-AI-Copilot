from pathlib import Path

import pytest
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


# 场景：验证Collection已存在，删除后重建Collection
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


# 场景：测试Collection不存在，delete_collection方法没被调用。
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


# 验证 Named Vectors 和 Payload 是否正确组装。
def test_build_hybrid_point_assembles_named_vectors(monkeypatch):
    fake_sparse_vector = models.SparseVector(
        indices=[100],
        values=[1.0],
    )

    monkeypatch.setattr(
        hybrid_index_builder,
        "encode_sparse",
        lambda _: fake_sparse_vector,
    )

    metadata = {
        "doc_id": "smt-alarm-sop",
        "department": "production",
    }

    original_metadata = metadata.copy()

    dense_vector = [0.1] * hybrid_index_builder.QDRANT_VECTOR_SIZE

    result = hybrid_index_builder.build_hybrid_point(
        point_id=1,
        text="贴片机出现 E203 报警",
        metadata=metadata,
        dense_vector=dense_vector,
    )

    assert result.id == 1
    assert result.vector[hybrid_index_builder.DENSE_VECTOR_NAME] == dense_vector
    assert result.vector[hybrid_index_builder.SPARSE_VECTOR_NAME] == fake_sparse_vector
    assert result.payload == {
        "doc_id": "smt-alarm-sop",
        "department": "production",
        "text": "贴片机出现 E203 报警",
    }
    assert metadata == original_metadata


# 传入非1024维 Dense vector 时应抛出 ValueError
def test_build_hybrid_point_rejects_wrong_dense_dimension():
    invalid_dense_vector = [0.1] * 768

    with pytest.raises(
        ValueError,
        match="Dense vector dimension mismatch",
    ):
        hybrid_index_builder.build_hybrid_point(
            point_id=1,
            text="贴片机出现 E203 报警",
            metadata={},
            dense_vector=invalid_dense_vector,
        )


# 验证hybrid Node的传入参数以及返回值是否正确传递
def test_build_hybrid_nodes_reuses_baseline_chunk_settings(monkeypatch):
    raw_dir = Path("data/raw")
    fake_documents = [object()]
    fake_nodes = [object()]
    received = {}

    def fake_load_markdown_documents(received_raw_dir):
        received["raw_dir"] = received_raw_dir
        return fake_documents

    class FakeSentenceSplitter:
        def __init__(self, *, chunk_size, chunk_overlap):
            received["chunk_size"] = chunk_size
            received["chunk_overlap"] = chunk_overlap

        def get_nodes_from_documents(self, documents):
            received["documents"] = documents
            return fake_nodes

    monkeypatch.setattr(
        hybrid_index_builder,
        "load_markdown_documents",
        fake_load_markdown_documents,
    )
    monkeypatch.setattr(
        hybrid_index_builder,
        "SentenceSplitter",
        FakeSentenceSplitter,
    )

    result = hybrid_index_builder.build_hybrid_nodes(raw_dir)

    assert received["raw_dir"] == raw_dir
    assert received["chunk_size"] == hybrid_index_builder.DEFAULT_CHUNK_SIZE
    assert received["chunk_overlap"] == hybrid_index_builder.DEFAULT_CHUNK_OVERLAP
    assert received["documents"] is fake_documents
    assert result is fake_nodes


# 验证传入空Nodes时，不初始化embedding模型。
def test_embed_hybrid_nodes_returns_empty_without_configuring(monkeypatch):
    configured = False

    def fake_configure_embedding():
        nonlocal configured
        configured = True

    monkeypatch.setattr(
        hybrid_index_builder,
        "configure_embedding",
        fake_configure_embedding,
    )

    result = hybrid_index_builder.embed_hybrid_nodes([])

    assert result == []
    assert configured is False


# 传入Node
def test_embed_hybrid_nodes_embeds_node_content_in_batch(monkeypatch):
    received = {
        "metadata_modes": [],
    }
    configured = False
    fake_vectors = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    class FakeNode:
        def __init__(self, content):
            self.content = content

        def get_content(self, *, metadata_mode):
            received["metadata_modes"].append(metadata_mode)
            return self.content

    class FakeEmbeddingModel:
        def get_text_embedding_batch(self, texts):
            received["texts"] = texts
            return fake_vectors

    class FakeSettings:
        embed_model = FakeEmbeddingModel()

    def fake_configure_embedding():
        nonlocal configured
        configured = True

    monkeypatch.setattr(
        hybrid_index_builder,
        "configure_embedding",
        fake_configure_embedding,
    )
    monkeypatch.setattr(
        hybrid_index_builder,
        "Settings",
        FakeSettings(),
    )

    nodes = [
        FakeNode("E203 报警处理"),
        FakeNode("检查送料器通信线"),
    ]

    result = hybrid_index_builder.embed_hybrid_nodes(nodes)

    assert configured is True
    assert received["metadata_modes"] == [
        hybrid_index_builder.MetadataMode.EMBED,
        hybrid_index_builder.MetadataMode.EMBED,
    ]
    assert received["texts"] == [
        "E203 报警处理",
        "检查送料器通信线",
    ]
    assert result is fake_vectors
