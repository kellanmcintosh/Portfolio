from unittest.mock import patch, MagicMock
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
        # Unique words so we can verify overlap precisely
        words = [f"word{i}" for i in range(600)]
        chunks = ingest.chunk_text(" ".join(words))
        assert len(chunks) >= 2
        end_of_first = set(chunks[0].split()[-10:])
        start_of_second = set(chunks[1].split()[:10])
        assert end_of_first & start_of_second, "Expected token overlap between consecutive chunks"

    def test_all_tokens_are_covered(self):
        # Every token in the source should appear in at least one chunk
        text = " ".join([f"tok{i}" for i in range(300)])
        chunks = ingest.chunk_text(text)
        combined = " ".join(chunks)
        for i in range(300):
            assert f"tok{i}" in combined


class TestEmbed:
    def _make_mock_result(self, vectors: list[list[float]]):
        embeddings = []
        for v in vectors:
            e = MagicMock()
            e.values = v
            embeddings.append(e)
        result = MagicMock()
        result.embeddings = embeddings
        return result

    def test_returns_one_vector_per_input(self):
        mock_result = self._make_mock_result([[0.1, 0.2], [0.3, 0.4]])
        with patch.object(ingest.client.models, "embed_content", return_value=mock_result):
            result = ingest.embed(["chunk one", "chunk two"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_calls_api_with_correct_model(self):
        mock_result = self._make_mock_result([[0.1]])
        with patch.object(ingest.client.models, "embed_content", return_value=mock_result) as mock_call:
            ingest.embed(["test"])
            assert mock_call.call_args.kwargs["model"] == ingest.EMBED_MODEL

    def test_calls_api_with_retrieval_document_task_type(self):
        mock_result = self._make_mock_result([[0.1]])
        with patch.object(ingest.client.models, "embed_content", return_value=mock_result) as mock_call:
            ingest.embed(["test"])
            config = mock_call.call_args.kwargs["config"]
            assert config.task_type == "RETRIEVAL_DOCUMENT"
