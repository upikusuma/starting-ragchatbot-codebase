"""Shared fixtures for backend tests.

backend/app.py builds a RAGSystem and mounts ../frontend at import time, which
fails in the test environment. The fixtures here build a lightweight test app
that mirrors the real API routes against a mocked RAGSystem.
"""
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    text: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(rag_system) -> FastAPI:
    """Build an app with the same routes as backend/app.py, minus static files."""
    app = FastAPI(title="Course Materials RAG System (test)")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}

    return app


@pytest.fixture
def sample_sources():
    return [
        {"text": "Intro to MCP - Lesson 1", "url": "https://example.com/mcp/1"},
        {"text": "Intro to MCP - Lesson 2", "url": None},
    ]


@pytest.fixture
def sample_courses():
    return {
        "total_courses": 2,
        "course_titles": ["Intro to MCP", "Advanced Retrieval"],
    }


@pytest.fixture
def mock_rag_system(sample_sources, sample_courses):
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "session_1"
    rag.query.return_value = ("MCP is a protocol.", sample_sources)
    rag.get_course_analytics.return_value = sample_courses
    return rag


@pytest.fixture
def client(mock_rag_system):
    return TestClient(create_test_app(mock_rag_system))
