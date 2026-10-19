"""Shared fixtures for the RAG chatbot tests.

Test layers (select with -m):
  default      unit tests with mocks, fast
  integration  real ChromaDB in a temp dir + the embedding model
  live         real Ollama server + a copy of the populated chroma_db
               (point RAG_LIVE_DB at a db, or it uses backend/chroma_db)
"""
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

# Normally set by pyproject's pytest `pythonpath`; kept so `pytest` also works from backend/
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import Course, CourseChunk, Lesson  # noqa: E402
from vector_store import SearchResults, VectorStore  # noqa: E402


# ---------------------------------------------------------------- helpers

def make_results(docs, metas):
    """Build SearchResults the way VectorStore.search would"""
    return SearchResults(documents=list(docs), metadata=list(metas), distances=[0.1] * len(docs))


def chunk_meta(course_title, lesson_number, chunk_index=0):
    return {"course_title": course_title, "lesson_number": lesson_number, "chunk_index": chunk_index}


def tool_call(name, **arguments):
    """Mimic an Ollama tool call: call.function.name / call.function.arguments"""
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def reply(content="", tool_calls=None):
    """Mimic an Ollama chat response: response.message.content / .tool_calls"""
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))


class FakeOllamaClient:
    """Scripted stand-in for ollama.Client that records every chat() call"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        recorded = dict(kwargs)
        recorded["messages"] = list(kwargs["messages"])  # snapshot, the caller may append later
        self.calls.append(recorded)
        return self.responses.pop(0)


# ---------------------------------------------------------------- unit fixtures

@pytest.fixture
def mock_store():
    """VectorStore double: search returns one chunk by default, lesson links resolve"""
    store = MagicMock(spec=VectorStore)
    store.search.return_value = make_results(
        ["Embeddings are lists of numbers."], [chunk_meta("Course A", 1)]
    )
    store.get_lesson_link.return_value = "https://example.com/a/lesson-1"
    return store


# ---------------------------------------------------------------- integration fixtures

COURSE_A = "Intro to Vector Databases"
COURSE_B = "Cooking Pasta From Scratch"


@pytest.fixture(scope="session")
def seeded_store(tmp_path_factory):
    """Real VectorStore on a temp ChromaDB with two small, unrelated courses"""
    store = VectorStore(str(tmp_path_factory.mktemp("chroma")), "all-MiniLM-L6-v2", max_results=5)

    courses = {
        COURSE_A: Course(
            title=COURSE_A,
            course_link="https://example.com/vdb",
            instructor="Ada",
            lessons=[
                Lesson(lesson_number=0, title="Welcome", lesson_link="https://example.com/vdb/lesson-0"),
                Lesson(lesson_number=1, title="Embeddings", lesson_link="https://example.com/vdb/lesson-1"),
                Lesson(lesson_number=2, title="Similarity Search", lesson_link="https://example.com/vdb/lesson-2"),
            ],
        ),
        COURSE_B: Course(
            title=COURSE_B,
            course_link="https://example.com/pasta",
            instructor="Luigi",
            lessons=[
                Lesson(lesson_number=1, title="Dough", lesson_link="https://example.com/pasta/lesson-1"),
                Lesson(lesson_number=2, title="Sauces", lesson_link="https://example.com/pasta/lesson-2"),
            ],
        ),
    }
    texts = {
        COURSE_A: [
            (0, "Welcome to the course on vector databases. We will learn how to store and query embeddings."),
            (1, "An embedding is a list of numbers that represents the meaning of text. "
                "Embeddings are stored in a vector database so that similar items sit close together."),
            (2, "Similarity search finds the nearest vectors to a query vector using cosine distance."),
        ],
        COURSE_B: [
            (1, "To make fresh pasta dough, mix flour and eggs, then knead for ten minutes and let it rest."),
            (2, "A simple tomato sauce simmers garlic, tomatoes and basil in olive oil for twenty minutes."),
        ],
    }
    for title, course in courses.items():
        store.add_course_metadata(course)
        store.add_course_content([
            CourseChunk(content=text, course_title=title, lesson_number=lesson, chunk_index=i)
            for i, (lesson, text) in enumerate(texts[title])
        ])
    return store


# ---------------------------------------------------------------- live fixtures

def _ollama_up(host):
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def live_config():
    """Config pointing at a throwaway copy of the populated DB; skips if Ollama or the DB is missing"""
    from config import Config

    cfg = Config()
    if not _ollama_up(cfg.OLLAMA_HOST):
        pytest.skip(f"Ollama not reachable at {cfg.OLLAMA_HOST}")
    source = Path(os.environ.get("RAG_LIVE_DB", BACKEND_DIR / "chroma_db"))
    if not source.exists():
        pytest.skip(f"No populated ChromaDB at {source} (set RAG_LIVE_DB)")
    target = Path(os.environ.get("TEMP", "/tmp")) / "rag_live_db_copy"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)  # never touch the real DB from tests
    cfg.CHROMA_PATH = str(target)
    return cfg
