"""Tests for how RAGSystem handles content-related questions end to end"""
from unittest.mock import MagicMock

import pytest

import rag_system
from config import Config
from conftest import COURSE_A, chunk_meta, make_results
from rag_system import RAGSystem


@pytest.fixture
def rag(monkeypatch, mock_store):
    """RAGSystem with VectorStore and AIGenerator replaced by doubles"""
    monkeypatch.setattr(rag_system, "VectorStore", lambda *a, **k: mock_store)
    monkeypatch.setattr(rag_system, "AIGenerator", lambda *a, **k: MagicMock())
    return RAGSystem(Config())


def search_then_answer(query="x", **search_args):
    """generate_response stand-in: a model that searches once, then answers with the tool output"""
    def generate(**kwargs):
        result = kwargs["tool_manager"].execute_tool("search_course_content", query=query, **search_args)
        return f"ANSWER<{result}>"
    return generate


# ------------------------------------------------------------ unit: orchestration

class TestOrchestration:
    def test_search_tool_is_registered(self, rag):
        names = [d["name"] for d in rag.tool_manager.get_tool_definitions()]
        assert "search_course_content" in names

    def test_query_hands_tools_and_manager_to_the_generator(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.query("What is MCP?")
        kwargs = rag.ai_generator.generate_response.call_args.kwargs
        assert kwargs["query"] == "What is MCP?"
        assert kwargs["tool_manager"] is rag.tool_manager
        assert [t["name"] for t in kwargs["tools"]] == ["search_course_content"]

    def test_query_returns_answer_and_sources_from_the_search(self, rag, mock_store):
        rag.ai_generator.generate_response.side_effect = search_then_answer()
        answer, sources = rag.query("What are embeddings?")
        assert answer.startswith("ANSWER<[Course A - Lesson 1]")
        assert sources == [{"text": "Course A - Lesson 1", "url": "https://example.com/a/lesson-1"}]

    def test_sources_do_not_leak_into_the_next_query(self, rag):
        rag.ai_generator.generate_response.side_effect = search_then_answer()
        rag.query("first")
        rag.ai_generator.generate_response.side_effect = lambda **kw: "no search this time"
        _, sources = rag.query("second")
        assert sources == []

    def test_empty_search_gives_no_sources_and_passes_the_message_on(self, rag, mock_store):
        mock_store.search.return_value = make_results([], [])
        rag.ai_generator.generate_response.side_effect = search_then_answer(course_name="MCP")
        answer, sources = rag.query("anything")
        assert "No relevant content found in course 'MCP'" in answer
        assert sources == []

    def test_conversation_history_is_stored_and_replayed(self, rag):
        rag.ai_generator.generate_response.return_value = "first answer"
        session = rag.session_manager.create_session()
        rag.query("first question", session_id=session)
        rag.query("second question", session_id=session)
        history = rag.ai_generator.generate_response.call_args.kwargs["conversation_history"]
        assert "User: first question" in history and "Assistant: first answer" in history

    def test_no_session_means_no_history(self, rag):
        rag.ai_generator.generate_response.return_value = "answer"
        rag.query("q")
        assert rag.ai_generator.generate_response.call_args.kwargs["conversation_history"] is None


# ------------------------------------------------------------ integration: real store, scripted model

@pytest.mark.integration
class TestWithRealStore:
    @pytest.fixture
    def rag_real(self, monkeypatch, seeded_store):
        monkeypatch.setattr(rag_system, "VectorStore", lambda *a, **k: seeded_store)
        monkeypatch.setattr(rag_system, "AIGenerator", lambda *a, **k: MagicMock())
        return RAGSystem(Config())

    def test_content_question_returns_chunk_text_and_lesson_link(self, rag_real):
        rag_real.ai_generator.generate_response.side_effect = search_then_answer(
            query="how are embeddings stored", course_name="Vector", lesson_number=1
        )
        answer, sources = rag_real.query("How are embeddings stored?")
        assert "embedding is a list of numbers" in answer
        assert sources == [{"text": f"{COURSE_A} - Lesson 1", "url": "https://example.com/vdb/lesson-1"}]

    def test_question_with_nonexistent_lesson_reports_no_content(self, rag_real):
        rag_real.ai_generator.generate_response.side_effect = search_then_answer(
            query="x", course_name="Vector", lesson_number=42
        )
        answer, sources = rag_real.query("What is in lesson 42?")
        assert "No relevant content found" in answer
        assert sources == []


# ------------------------------------------------------------ live: real model + real DB copy

LIVE_QUESTIONS = [
    "What is MCP and why is it useful?",
    "What does lesson 2 of the Computer Use course cover?",
    "How does prompt caching work in the Advanced Retrieval course?",
    "Explain what a vector database is according to the courses",
]


@pytest.mark.live
class TestLiveEndToEnd:
    @pytest.fixture(scope="class")
    def live_rag(self, live_config):
        return RAGSystem(live_config)

    @pytest.mark.parametrize("question", LIVE_QUESTIONS)
    def test_content_question_is_answered_from_search_results(self, live_rag, question, monkeypatch):
        outputs = []
        original = live_rag.tool_manager.execute_tool

        def spy(name, **kwargs):
            outputs.append(original(name, **kwargs))
            return outputs[-1]

        monkeypatch.setattr(live_rag.tool_manager, "execute_tool", spy)
        answer, sources = live_rag.query(question)

        assert outputs, f"search tool was never called; answer: {answer[:120]!r}"
        assert not outputs[0].startswith("No relevant content found"), outputs[0]
        assert answer.strip()
        assert sources, "answer came back without any sources"
