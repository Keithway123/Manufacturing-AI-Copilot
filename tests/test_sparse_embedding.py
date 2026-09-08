from manufacturing_ai_copilot.rag import sparse_embedding


# 1.英文字母统一为小写
# 2.过滤空白token
# 3.过滤纯标点token
def test_tokenize_normalizes_case_and_removes_punctuation():
    tokens = sparse_embedding.tokenize("E203！")

    assert tokens == ["e203"]


# 验证encode_sparse的转换逻辑
def test_encode_sparse_converts_token_counts_to_weights(monkeypatch):
    monkeypatch.setattr(
        sparse_embedding,
        "tokenize",
        lambda text: ["设备", "设备", "e203"],
    )

    token_indices = {
        "设备": 100,
        "e203": 200,
    }

    monkeypatch.setattr(
        sparse_embedding,
        "token_to_index",
        lambda token: token_indices[token],
    )

    # ignored 只是随便的一个参数,真实参数是lambda text
    result = sparse_embedding.encode_sparse("ignored")

    assert result.indices == [100, 200]
    assert result.values == [2.0, 1.0]
