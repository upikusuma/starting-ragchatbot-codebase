"""API tests for /api/query, /api/courses and /."""

import pytest


class TestQueryEndpoint:
    def test_creates_session_when_missing(self, client, mock_rag_system):
        res = client.post("/api/query", json={"query": "What is MCP?"})

        assert res.status_code == 200
        body = res.json()
        assert body["answer"] == "MCP is a protocol."
        assert body["session_id"] == "session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "session_1")

    def test_reuses_provided_session(self, client, mock_rag_system):
        res = client.post("/api/query", json={"query": "Hi", "session_id": "abc"})

        assert res.status_code == 200
        assert res.json()["session_id"] == "abc"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Hi", "abc")

    def test_returns_sources_with_optional_urls(self, client, sample_sources):
        sources = client.post("/api/query", json={"query": "x"}).json()["sources"]

        assert sources == sample_sources

    @pytest.mark.parametrize(
        "payload",
        [{}, {"session_id": "abc"}, {"query": 123}, {"query": None}],
    )
    def test_invalid_payload_returns_422(self, client, payload):
        assert client.post("/api/query", json=payload).status_code == 422

    def test_non_json_body_returns_422(self, client):
        res = client.post(
            "/api/query",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 422

    def test_rag_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("model offline")

        res = client.post("/api/query", json={"query": "x"})

        assert res.status_code == 500
        assert res.json()["detail"] == "model offline"

    def test_get_not_allowed(self, client):
        assert client.get("/api/query").status_code == 405


class TestCoursesEndpoint:
    def test_returns_stats(self, client, sample_courses):
        res = client.get("/api/courses")

        assert res.status_code == 200
        assert res.json() == sample_courses

    def test_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        assert client.get("/api/courses").json() == {
            "total_courses": 0,
            "course_titles": [],
        }

    def test_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("db down")

        res = client.get("/api/courses")

        assert res.status_code == 500
        assert res.json()["detail"] == "db down"

    def test_post_not_allowed(self, client):
        assert client.post("/api/courses").status_code == 405


class TestRootEndpoint:
    def test_root_responds(self, client):
        res = client.get("/")

        assert res.status_code == 200
        assert "message" in res.json()

    def test_unknown_route_404(self, client):
        assert client.get("/api/nope").status_code == 404
