from unittest.mock import MagicMock, patch

import ingest


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        result = ingest.chunk_text("hello world")
        assert len(result) == 1
        assert "hello" in result[0]

    def test_empty_text_returns_no_chunks(self):
        result = ingest.chunk_text("")
        assert result == []

    def test_long_text_creates_multiple_chunks(self):
        long_text = " ".join(["word"] * 600)
        result = ingest.chunk_text(long_text)
        assert len(result) > 1

    def test_chunk_size_never_exceeds_limit(self):
        long_text = " ".join(["word"] * 1000)
        for chunk in ingest.chunk_text(long_text):
            assert len(ingest.tokenizer.encode(chunk)) <= ingest.CHUNK_TOKENS

    def test_overlap_produces_shared_content(self):
        words = [f"word{i}" for i in range(600)]
        chunks = ingest.chunk_text(" ".join(words))
        assert len(chunks) >= 2
        # 50-token overlap ≈ 25-50 words depending on tokenization; use 60-word windows to be safe
        end_of_first = set(chunks[0].split()[-60:])
        start_of_second = set(chunks[1].split()[:60])
        assert end_of_first & start_of_second, "Expected token overlap between consecutive chunks"

    def test_all_tokens_are_covered(self):
        text = " ".join([f"tok{i}" for i in range(300)])
        chunks = ingest.chunk_text(text)
        combined = " ".join(chunks)
        for i in range(300):
            assert f"tok{i}" in combined


class TestEmbed:
    def _mock_encode(self, vectors: list[list[float]]) -> MagicMock:
        result = MagicMock()
        result.tolist.return_value = vectors
        return result

    def test_returns_one_vector_per_input(self):
        with patch.object(ingest, "embed_model") as mock_model:
            mock_model.encode.return_value = self._mock_encode([[0.1, 0.2], [0.3, 0.4]])
            result = ingest.embed(["chunk one", "chunk two"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_passes_all_texts_in_one_call(self):
        with patch.object(ingest, "embed_model") as mock_model:
            mock_model.encode.return_value = self._mock_encode([[0.1], [0.2], [0.3]])
            ingest.embed(["a", "b", "c"])
            texts_arg = mock_model.encode.call_args.args[0]
            assert texts_arg == ["a", "b", "c"]
            assert mock_model.encode.call_count == 1

    def test_uses_normalized_embeddings(self):
        with patch.object(ingest, "embed_model") as mock_model:
            mock_model.encode.return_value = self._mock_encode([[0.1]])
            ingest.embed(["test"])
            kwargs = mock_model.encode.call_args.kwargs
            assert kwargs.get("normalize_embeddings") is True
