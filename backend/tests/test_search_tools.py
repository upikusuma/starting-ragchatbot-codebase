"""Tests for CourseSearchTool.execute and the ToolManager that exposes it"""
import os

import pytest

from conftest import (
    BACKEND_DIR, COURSE_A, COURSE_B, REPO_ROOT, chunk_meta, make_results,
)
from search_tools import CourseSearchTool, ToolManager


# ------------------------------------------------------------ unit: mocked store

class TestExecuteWithMockedStore:
    def test_store_error_is_returned_verbatim(self, mock_store):
        mock_store.search.return_value = make_results([], [])
        mock_store.search.return_value.error = "Search error: boom"
        assert CourseSearchTool(mock_store).execute(query="x") == "Search error: boom"

    def test_empty_results_message_without_filters(self, mock_store):
        mock_store.search.return_value = make_results([], [])
        assert CourseSearchTool(mock_store).execute(query="x") == "No relevant content found."

    def test_empty_results_message_names_course_and_lesson(self, mock_store):
        mock_store.search.return_value = make_results([], [])
        out = CourseSearchTool(mock_store).execute(query="x", course_name="MCP", lesson_number=3)
        assert out == "No relevant content found in course 'MCP' in lesson 3."

    def test_empty_results_message_mentions_lesson_zero(self, mock_store):
        """Lesson 0 is a real lesson (the intro); the message must not silently drop it"""
        mock_store.search.return_value = make_results([], [])
        out = CourseSearchTool(mock_store).execute(query="x", lesson_number=0)
        assert "lesson 0" in out

    def test_arguments_are_forwarded_to_the_store(self, mock_store):
        CourseSearchTool(mock_store).execute(query="caching", course_name="MCP", lesson_number=2)
        mock_store.search.assert_called_once_with(query="caching", course_name="MCP", lesson_number=2)

    def test_optional_arguments_default_to_none(self, mock_store):
        CourseSearchTool(mock_store).execute(query="caching")
        mock_store.search.assert_called_once_with(query="caching", course_name=None, lesson_number=None)

    def test_results_are_formatted_with_course_and_lesson_header(self, mock_store):
        out = CourseSearchTool(mock_store).execute(query="x")
        assert out == "[Course A - Lesson 1]\nEmbeddings are lists of numbers."

    def test_chunk_without_lesson_number_has_course_only_header(self, mock_store):
        mock_store.search.return_value = make_results(["doc"], [chunk_meta("Course A", None)])
        assert CourseSearchTool(mock_store).execute(query="x").startswith("[Course A]\n")

    def test_multiple_results_are_separated_by_blank_line(self, mock_store):
        mock_store.search.return_value = make_results(
            ["one", "two"], [chunk_meta("Course A", 1), chunk_meta("Course A", 2)]
        )
        out = CourseSearchTool(mock_store).execute(query="x")
        assert out == "[Course A - Lesson 1]\none\n\n[Course A - Lesson 2]\ntwo"

    def test_sources_carry_text_and_lesson_link(self, mock_store):
        tool = CourseSearchTool(mock_store)
        tool.execute(query="x")
        assert tool.last_sources == [{"text": "Course A - Lesson 1", "url": "https://example.com/a/lesson-1"}]
        mock_store.get_lesson_link.assert_called_once_with("Course A", 1)

    def test_sources_are_deduplicated(self, mock_store):
        mock_store.search.return_value = make_results(
            ["one", "two"], [chunk_meta("Course A", 1, 0), chunk_meta("Course A", 1, 1)]
        )
        tool = CourseSearchTool(mock_store)
        tool.execute(query="x")
        assert len(tool.last_sources) == 1

    def test_source_without_lesson_has_no_url(self, mock_store):
        mock_store.search.return_value = make_results(["doc"], [chunk_meta("Course A", None)])
        tool = CourseSearchTool(mock_store)
        tool.execute(query="x")
        assert tool.last_sources == [{"text": "Course A", "url": None}]
        mock_store.get_lesson_link.assert_not_called()

    def test_tool_definition_matches_what_the_model_is_told(self, mock_store):
        definition = CourseSearchTool(mock_store).get_tool_definition()
        assert definition["name"] == "search_course_content"
        assert definition["input_schema"]["required"] == ["query"]
        assert set(definition["input_schema"]["properties"]) == {"query", "course_name", "lesson_number"}


class TestToolManager:
    def test_execute_tool_dispatches_by_name(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        out = manager.execute_tool("search_course_content", query="x")
        assert out.startswith("[Course A - Lesson 1]")

    def test_unknown_tool_is_reported_not_raised(self):
        assert ToolManager().execute_tool("nope") == "Tool 'nope' not found"

    def test_sources_can_be_read_then_reset(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        manager.execute_tool("search_course_content", query="x")
        assert manager.get_last_sources()
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_wrong_argument_name_raises_type_error(self, mock_store):
        """A model that invents a parameter (e.g. 'topic') crashes execute; the caller must catch it"""
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        with pytest.raises(TypeError):
            manager.execute_tool("search_course_content", query="x", topic="y")


# ------------------------------------------------------------ integration: real ChromaDB

@pytest.mark.integration
class TestExecuteWithRealStore:
    def test_unfiltered_query_finds_the_relevant_course(self, seeded_store):
        out = CourseSearchTool(seeded_store).execute(query="How are embeddings stored?")
        assert out.startswith(f"[{COURSE_A}")

    def test_unfiltered_query_finds_other_course_too(self, seeded_store):
        out = CourseSearchTool(seeded_store).execute(query="How do I make pasta dough?")
        assert out.startswith(f"[{COURSE_B}")

    def test_partial_course_name_filters_to_that_course(self, seeded_store):
        out = CourseSearchTool(seeded_store).execute(query="how do I start", course_name="Vector")
        assert COURSE_B not in out
        assert f"[{COURSE_A}" in out

    def test_course_and_lesson_filter_returns_only_that_lesson(self, seeded_store):
        out = CourseSearchTool(seeded_store).execute(query="sauce", course_name="Pasta", lesson_number=2)
        assert out.startswith(f"[{COURSE_B} - Lesson 2]")
        assert "Lesson 1" not in out

    def test_lesson_zero_is_searchable(self, seeded_store):
        out = CourseSearchTool(seeded_store).execute(query="welcome", course_name="Vector", lesson_number=0)
        assert f"[{COURSE_A} - Lesson 0]" in out

    def test_sources_link_to_the_lesson(self, seeded_store):
        tool = CourseSearchTool(seeded_store)
        tool.execute(query="embeddings", course_name="Vector", lesson_number=1)
        assert tool.last_sources == [
            {"text": f"{COURSE_A} - Lesson 1", "url": "https://example.com/vdb/lesson-1"}
        ]

    def test_lesson_that_does_not_exist_gives_informative_empty_message(self, seeded_store):
        out = CourseSearchTool(seeded_store).execute(query="x", course_name="Vector", lesson_number=99)
        assert out == "No relevant content found in course 'Vector' in lesson 99."

    def test_unrelated_course_name_is_not_forced_onto_a_random_course(self, seeded_store):
        """_resolve_course_name returns the nearest course whatever the distance, so
        'Quantum Astrophysics' silently filters every search to one unrelated course."""
        assert seeded_store._resolve_course_name("Quantum Astrophysics") is None

    def test_vague_course_name_does_not_hide_the_right_answer(self, seeded_store):
        """Model sends a loose course_name; the answer should still come from the right course"""
        out = CourseSearchTool(seeded_store).execute(query="nearest vectors cosine distance",
                                                     course_name="databases course")
        assert out.startswith(f"[{COURSE_A}")


# ------------------------------------------------------------ environment: where does the DB live?

class TestEnvironment:
    def test_chroma_path_does_not_depend_on_launch_directory(self, monkeypatch):
        """./chroma_db resolves differently from backend/ and from the repo root, so a server
        started from the root reads an empty DB and every search says 'No relevant content'."""
        from config import Config

        resolved = set()
        for cwd in (BACKEND_DIR, REPO_ROOT):
            monkeypatch.chdir(cwd)
            resolved.add(os.path.abspath(Config().CHROMA_PATH))
        assert len(resolved) == 1, f"CHROMA_PATH resolves to different places: {sorted(resolved)}"

    def test_docs_path_does_not_depend_on_launch_directory(self):
        """app.py loads documents from a cwd-relative '../docs' on startup"""
        source = (BACKEND_DIR / "app.py").read_text(encoding="utf-8")
        assert '"../docs"' not in source and "'../docs'" not in source
