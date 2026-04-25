from unittest.mock import patch, MagicMock
import app


class TestBuildPrompt:
    def test_contains_question(self):
        prompt = app.build_prompt("What is the policy?", ["some context"], ["doc.pdf"])
        assert "What is the policy?" in prompt

    def test_contains_chunk_text(self):
        prompt = app.build_prompt("Q?", ["The answer is 42"], ["doc.pdf"])
        assert "The answer is 42" in prompt

    def test_contains_source_label(self):
        prompt = app.build_prompt("Q?", ["chunk"], ["report.pdf"])
        assert "report.pdf" in prompt

    def test_multiple_chunks_all_present(self):
        prompt = app.build_prompt("Q?", ["chunk A", "chunk B"], ["a.pdf", "b.pdf"])
        assert "chunk A" in prompt
        assert "chunk B" in prompt
        assert "a.pdf" in prompt
        assert "b.pdf" in prompt

    def test_instructs_model_to_use_only_context(self):
        prompt = app.build_prompt("Q?", ["chunk"], ["doc.pdf"])
        assert "only the context" in prompt

    def test_includes_fallback_instruction(self):
        prompt = app.build_prompt("Q?", ["chunk"], ["doc.pdf"])
        assert "don't know" in prompt


class TestEmbedQuery:
    def _mock_response(self, vector: list[float]) -> MagicMock:
        mock = MagicMock()
        mock.json.return_value = {"embedding": {"values": vector}}
        return mock

    def test_returns_vector(self):
        with patch("app.requests.post", return_value=self._mock_response([0.1, 0.2, 0.3])):
            result = app.embed_query("What is X?")
        assert result == [0.1, 0.2, 0.3]

    def test_hits_v1_stable_endpoint(self):
        with patch("app.requests.post", return_value=self._mock_response([0.1])) as mock_post:
            app.embed_query("test")
            url = mock_post.call_args.args[0]
            assert "/v1/" in url
            assert "v1beta" not in url

    def test_sends_retrieval_query_task_type(self):
        with patch("app.requests.post", return_value=self._mock_response([0.1])) as mock_post:
            app.embed_query("test")
            body = mock_post.call_args.kwargs["json"]
            assert body["taskType"] == "RETRIEVAL_QUERY"

    def test_uses_correct_model(self):
        with patch("app.requests.post", return_value=self._mock_response([0.1])) as mock_post:
            app.embed_query("test")
            body = mock_post.call_args.kwargs["json"]
            assert app.EMBED_MODEL in body["model"]


class TestRetrieve:
    def _mock_collection(self, chunks, sources):
        col = MagicMock()
        col.query.return_value = {
            "documents": [chunks],
            "metadatas": [[{"source": s, "chunk_index": i} for i, s in enumerate(sources)]],
        }
        return col

    def test_returns_chunks_and_sources(self):
        col = self._mock_collection(["chunk one", "chunk two"], ["a.pdf", "b.pdf"])
        with patch.object(app, "embed_query", return_value=[0.1, 0.2]):
            chunks, sources = app.retrieve(col, "question")
        assert chunks == ["chunk one", "chunk two"]
        assert sources == ["a.pdf", "b.pdf"]

    def test_queries_top_k_results(self):
        col = self._mock_collection([], [])
        with patch.object(app, "embed_query", return_value=[0.1]):
            app.retrieve(col, "q")
        assert col.query.call_args.kwargs["n_results"] == app.TOP_K


class TestAsk:
    def _mock_generate_response(self, text: str) -> MagicMock:
        mock = MagicMock()
        mock.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": text}]}}]
        }
        return mock

    def _mock_collection(self, chunks, sources):
        col = MagicMock()
        col.query.return_value = {
            "documents": [chunks],
            "metadatas": [[{"source": s, "chunk_index": i} for i, s in enumerate(sources)]],
        }
        return col

    def test_returns_model_answer(self):
        col = self._mock_collection(["context"], ["doc.pdf"])
        with patch.object(app, "embed_query", return_value=[0.1]), \
             patch("app.requests.post", return_value=self._mock_generate_response("42 is the answer.")):
            answer, _ = app.ask(col, "What is the answer?")
        assert answer == "42 is the answer."

    def test_deduplicates_sources(self):
        col = self._mock_collection(["c1", "c2", "c3"], ["doc.pdf", "doc.pdf", "other.pdf"])
        with patch.object(app, "embed_query", return_value=[0.1]), \
             patch("app.requests.post", return_value=self._mock_generate_response("answer")):
            _, sources = app.ask(col, "question")
        assert sources.count("doc.pdf") == 1
        assert "other.pdf" in sources
