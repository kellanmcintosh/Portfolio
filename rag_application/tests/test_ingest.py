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
        words = [f"word{i}" for i in range(600)]
        chunks = ingest.chunk_text(" ".join(words))
        assert len(chunks) >= 2
        end_of_first = set(chunks[0].split()[-10:])
        start_of_second = set(chunks[1].split()[:10])
        assert end_of_first & start_of_second, "Expected token overlap between consecutive chunks"

    def test_all_tokens_are_covered(self):
        text = " ".join([f"tok{i}" for i in range(300)])
        chunks = ingest.chunk_text(text)
        combined = " ".join(chunks)
        for i in range(300):
            assert f"tok{i}" in combined


class TestEmbed:
    def _mock_response(self, vectors: list[list[float]]) -> MagicMock:
        mock = MagicMock()
        mock.json.return_value = {"embeddings": [{"values": v} for v in vectors]}
        return mock

    def test_returns_one_vector_per_input(self):
        with patch("ingest.requests.post", return_value=self._mock_response([[0.1, 0.2], [0.3, 0.4]])):
            result = ingest.embed(["chunk one", "chunk two"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_hits_v1_stable_endpoint(self):
        with patch("ingest.requests.post", return_value=self._mock_response([[0.1]])) as mock_post:
            ingest.embed(["test"])
            url = mock_post.call_args.args[0]
            assert "/v1/" in url
            assert "v1beta" not in url

    def test_uses_batch_embed_endpoint(self):
        with patch("ingest.requests.post", return_value=self._mock_response([[0.1]])) as mock_post:
            ingest.embed(["test"])
            url = mock_post.call_args.args[0]
            assert "batchEmbedContents" in url

    def test_sends_retrieval_document_task_type(self):
        with patch("ingest.requests.post", return_value=self._mock_response([[0.1]])) as mock_post:
            ingest.embed(["test"])
            body = mock_post.call_args.kwargs["json"]
            assert body["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"

    def test_batches_all_texts_in_one_request(self):
        texts = ["a", "b", "c"]
        with patch("ingest.requests.post", return_value=self._mock_response([[0.1]] * 3)) as mock_post:
            ingest.embed(texts)
            body = mock_post.call_args.kwargs["json"]
            assert len(body["requests"]) == 3
            assert mock_post.call_count == 1
