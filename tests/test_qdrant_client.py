from manufacturing_ai_copilot.rag import qdrant_client


def test_get_qdrant_client_uses_configured_url(monkeypatch):
    expected_url = "http://qdrant.test:6333"

    class FakeQdrantClient:
        def __init__(self, *, url: str):
            # 保存传入值，便于验证客户端工厂是否正确传递配置。
            self.url = url

    # 替换函数实际使用位置的配置和客户端，测试不访问真实网络。
    monkeypatch.setattr(qdrant_client, "QDRANT_URL", expected_url)
    monkeypatch.setattr(
        qdrant_client,
        "QdrantClient",
        FakeQdrantClient,
    )

    result = qdrant_client.get_qdrant_client()

    assert isinstance(result, FakeQdrantClient)
    assert result.url == expected_url
