"""Tests that AIGenerator asks for, executes, and feeds back CourseSearchTool correctly"""

from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator
from conftest import FakeOllamaClient, chunk_meta, make_results, reply, tool_call
from search_tools import CourseSearchTool, ToolManager


def make_generator(responses):
    generator = AIGenerator("fake-model", "http://localhost:11434")
    generator.client = FakeOllamaClient(responses)
    return generator


SEARCH_DEF = {
    "name": "search_course_content",
    "description": "Search course materials",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


# ------------------------------------------------------------ unit: request building


class TestRequest:
    def test_tool_definitions_are_converted_to_ollama_function_format(self):
        converted = AIGenerator._to_ollama_tools([SEARCH_DEF])
        assert converted == [
            {
                "type": "function",
                "function": {
                    "name": "search_course_content",
                    "description": "Search course materials",
                    "parameters": SEARCH_DEF["input_schema"],
                },
            }
        ]

    def test_tools_are_sent_when_provided(self):
        generator = make_generator([reply("hi")])
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=MagicMock())
        assert (
            generator.client.calls[0]["tools"][0]["function"]["name"]
            == "search_course_content"
        )

    def test_tools_are_omitted_when_not_provided(self):
        generator = make_generator([reply("hi")])
        generator.generate_response("q")
        assert "tools" not in generator.client.calls[0]

    def test_user_question_is_passed_unchanged(self):
        generator = make_generator([reply("hi")])
        generator.generate_response("What is MCP?")
        assert generator.client.calls[0]["messages"][-1] == {
            "role": "user",
            "content": "What is MCP?",
        }

    def test_conversation_history_goes_into_system_message(self):
        generator = make_generator([reply("hi")])
        generator.generate_response(
            "q", conversation_history="User: earlier\nAssistant: reply"
        )
        system = generator.client.calls[0]["messages"][0]
        assert system["role"] == "system"
        assert "User: earlier" in system["content"]

    def test_system_prompt_names_search_for_course_questions(self):
        prompt = AIGenerator.SYSTEM_PROMPT.lower()
        assert "search" in prompt and "course" in prompt

    def test_system_prompt_allows_sequential_tool_calls(self):
        prompt = AIGenerator.SYSTEM_PROMPT.lower()
        assert "one search per query" not in prompt
        assert "2 sequential" in prompt


# ------------------------------------------------------------ unit: tool-call handling


class TestToolCalls:
    def test_no_tool_call_returns_content_without_second_request(self):
        generator = make_generator([reply("direct answer")])
        manager = MagicMock()
        assert (
            generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
            == "direct answer"
        )
        assert len(generator.client.calls) == 1
        manager.execute_tool.assert_not_called()

    def test_tool_call_without_a_tool_manager_returns_the_empty_content(self):
        """If tools were offered but no manager was given, tool calls cannot run"""
        call = tool_call("search_course_content", query="x")
        generator = make_generator([reply("", [call])])
        assert generator.generate_response("q", tools=[SEARCH_DEF]) == ""

    def test_search_tool_is_executed_with_the_model_arguments(self):
        call = tool_call(
            "search_course_content", query="caching", course_name="MCP", lesson_number=2
        )
        generator = make_generator([reply("", [call]), reply("final")])
        manager = MagicMock()
        manager.execute_tool.return_value = "tool output"
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
        manager.execute_tool.assert_called_once_with(
            "search_course_content", query="caching", course_name="MCP", lesson_number=2
        )

    @pytest.mark.parametrize("empty", [None, "", "null"])
    def test_empty_argument_values_are_dropped(self, empty):
        call = tool_call(
            "search_course_content", query="x", course_name=empty, lesson_number=empty
        )
        generator = make_generator([reply("", [call]), reply("final")])
        manager = MagicMock()
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
        manager.execute_tool.assert_called_once_with("search_course_content", query="x")

    def test_string_lesson_number_is_coerced_to_int(self):
        call = tool_call("search_course_content", query="x", lesson_number="2")
        generator = make_generator([reply("", [call]), reply("final")])
        manager = MagicMock()
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
        manager.execute_tool.assert_called_once_with(
            "search_course_content", query="x", lesson_number=2
        )

    def test_unparseable_lesson_number_is_dropped(self):
        call = tool_call("search_course_content", query="x", lesson_number="two")
        generator = make_generator([reply("", [call]), reply("final")])
        manager = MagicMock()
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
        manager.execute_tool.assert_called_once_with("search_course_content", query="x")

    def test_tool_result_is_sent_back_and_answer_returned(self):
        call = tool_call("search_course_content", query="x")
        generator = make_generator([reply("", [call]), reply("final answer")])
        manager = MagicMock()
        manager.execute_tool.return_value = "RESULT TEXT"
        answer = generator.generate_response(
            "q", tools=[SEARCH_DEF], tool_manager=manager
        )

        assert answer == "final answer"
        assert len(generator.client.calls) == 2
        assert generator.client.calls[1]["messages"][-1] == {
            "role": "tool",
            "tool_name": "search_course_content",
            "content": "RESULT TEXT",
        }

    def test_tool_exception_is_reported_to_the_model_not_raised(self):
        call = tool_call("search_course_content", query="x")
        generator = make_generator([reply("", [call]), reply("final")])
        manager = MagicMock()
        manager.execute_tool.side_effect = TypeError("unexpected keyword 'topic'")
        assert (
            generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
            == "final"
        )
        assert generator.client.calls[1]["messages"][-1]["content"].startswith(
            "Tool error:"
        )

    def test_every_tool_call_in_one_response_is_executed(self):
        calls = [
            tool_call("search_course_content", query="a"),
            tool_call("search_course_content", query="b"),
        ]
        generator = make_generator([reply("", calls), reply("final")])
        manager = MagicMock()
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
        assert manager.execute_tool.call_count == 2


# ------------------------------------------------------------ sequential tool rounds

OUTLINE_DEF = {
    "name": "get_course_outline",
    "description": "Get a course outline",
    "input_schema": {
        "type": "object",
        "properties": {"course_name": {"type": "string"}},
        "required": ["course_name"],
    },
}


def tool_contents(request):
    return [
        m["content"]
        for m in request["messages"]
        if isinstance(m, dict) and m.get("role") == "tool"
    ]


class TestSequentialRounds:
    def outline_then_search(self):
        return [
            reply("", [tool_call("get_course_outline", course_name="X")]),
            reply("", [tool_call("search_course_content", query="Embeddings")]),
            reply("final answer"),
        ]

    def test_two_rounds_are_separate_requests_executed_in_order(self):
        generator = make_generator(self.outline_then_search())
        manager = MagicMock()
        manager.execute_tool.side_effect = [
            "Lesson 4: Embeddings",
            "Course B covers embeddings",
        ]

        answer = generator.generate_response(
            "q", tools=[SEARCH_DEF, OUTLINE_DEF], tool_manager=manager
        )

        assert answer == "final answer"
        assert len(generator.client.calls) == 3
        assert [c.args[0] for c in manager.execute_tool.call_args_list] == [
            "get_course_outline",
            "search_course_content",
        ]
        manager.execute_tool.assert_any_call(
            "search_course_content", query="Embeddings"
        )

    def test_tools_are_offered_on_both_rounds_but_not_on_the_final_call(self):
        generator = make_generator(self.outline_then_search())
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=MagicMock())
        calls = generator.client.calls
        assert "tools" in calls[0] and "tools" in calls[1]
        assert "tools" not in calls[2]

    def test_later_requests_carry_earlier_results_in_order(self):
        generator = make_generator(self.outline_then_search())
        manager = MagicMock()
        manager.execute_tool.side_effect = ["OUTLINE", "SEARCH"]
        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)

        calls = generator.client.calls
        assert tool_contents(calls[1]) == ["OUTLINE"]
        assert tool_contents(calls[2]) == ["OUTLINE", "SEARCH"]
        assert calls[2]["messages"][1] == {"role": "user", "content": "q"}

    def test_model_answering_after_one_round_stops_there(self):
        call = tool_call("search_course_content", query="x")
        generator = make_generator([reply("", [call]), reply("done")])
        manager = MagicMock()

        assert (
            generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)
            == "done"
        )
        assert len(generator.client.calls) == 2
        assert "tools" in generator.client.calls[1]
        manager.execute_tool.assert_called_once()

    def test_a_third_round_is_never_executed(self):
        def call():
            return reply("", [tool_call("search_course_content", query="again")])

        generator = make_generator(
            [call(), call(), call()]
        )  # final reply still asks for a tool
        manager = MagicMock()

        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)

        assert len(generator.client.calls) == 3
        assert manager.execute_tool.call_count == 2
        assert "tools" not in generator.client.calls[2]

    def test_parallel_calls_in_one_round_count_as_one_round(self):
        calls = [
            tool_call("search_course_content", query="a"),
            tool_call("search_course_content", query="b"),
        ]
        generator = make_generator(
            [
                reply("", calls),
                reply("", [tool_call("search_course_content", query="c")]),
                reply("final"),
            ]
        )
        manager = MagicMock()

        generator.generate_response("q", tools=[SEARCH_DEF], tool_manager=manager)

        assert manager.execute_tool.call_count == 3
        assert len(generator.client.calls) == 3

    def test_failure_in_first_round_skips_round_two_and_answers_without_tools(self):
        call = tool_call("search_course_content", query="x")
        generator = make_generator([reply("", [call]), reply("sorry, search failed")])
        manager = MagicMock()
        manager.execute_tool.side_effect = RuntimeError("db down")

        answer = generator.generate_response(
            "q", tools=[SEARCH_DEF], tool_manager=manager
        )

        assert answer == "sorry, search failed"
        assert len(generator.client.calls) == 2
        assert "tools" not in generator.client.calls[1]
        assert tool_contents(generator.client.calls[1]) == ["Tool error: db down"]

    def test_failure_in_second_round_keeps_first_result_and_stops(self):
        generator = make_generator(self.outline_then_search())
        manager = MagicMock()
        manager.execute_tool.side_effect = ["OUTLINE", RuntimeError("db down")]

        answer = generator.generate_response(
            "q", tools=[SEARCH_DEF], tool_manager=manager
        )

        assert answer == "final answer"
        assert len(generator.client.calls) == 3
        assert "tools" not in generator.client.calls[2]
        assert tool_contents(generator.client.calls[2]) == [
            "OUTLINE",
            "Tool error: db down",
        ]


# ------------------------------------------------------------ wiring: generator -> real ToolManager -> CourseSearchTool


class TestGeneratorToSearchTool:
    def test_model_tool_call_reaches_the_vector_store_search(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        call = tool_call(
            "search_course_content",
            query="embeddings",
            course_name="Vector",
            lesson_number="1",
        )
        generator = make_generator([reply("", [call]), reply("final")])

        generator.generate_response(
            "q", tools=manager.get_tool_definitions(), tool_manager=manager
        )

        mock_store.search.assert_called_once_with(
            query="embeddings", course_name="Vector", lesson_number=1
        )
        tool_message = generator.client.calls[1]["messages"][-1]
        assert tool_message["content"].startswith("[Course A - Lesson 1]")

    def test_search_results_text_is_what_the_model_sees(self, mock_store):
        mock_store.search.return_value = make_results([], [])
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        call = tool_call("search_course_content", query="x")
        generator = make_generator([reply("", [call]), reply("final")])

        generator.generate_response(
            "q", tools=manager.get_tool_definitions(), tool_manager=manager
        )

        assert (
            generator.client.calls[1]["messages"][-1]["content"]
            == "No relevant content found."
        )

    def test_two_searches_in_sequence_both_reach_the_vector_store(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        generator = make_generator(
            [
                reply("", [tool_call("search_course_content", query="first")]),
                reply(
                    "",
                    [
                        tool_call(
                            "search_course_content",
                            query="second",
                            course_name="Vector",
                        )
                    ],
                ),
                reply("final"),
            ]
        )

        generator.generate_response(
            "q", tools=manager.get_tool_definitions(), tool_manager=manager
        )

        assert [c.kwargs["query"] for c in mock_store.search.call_args_list] == [
            "first",
            "second",
        ]

    def test_invented_parameter_name_does_not_crash_the_request(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        call = tool_call(
            "search_course_content", topic="embeddings"
        )  # model ignored the schema
        generator = make_generator([reply("", [call]), reply("final")])

        assert (
            generator.generate_response(
                "q", tools=manager.get_tool_definitions(), tool_manager=manager
            )
            == "final"
        )
        assert "Tool error" in generator.client.calls[1]["messages"][-1]["content"]


# ------------------------------------------------------------ live: does the real model call the tool?

LIVE_QUESTIONS = [
    "What is MCP and why is it useful?",
    "What does lesson 2 of the Computer Use course cover?",
    "How does prompt caching work in the Advanced Retrieval course?",
    "Explain what a vector database is according to the courses",
]
GENERIC_QUERIES = {
    "cover",
    "covers",
    "course",
    "lesson",
    "content",
    "about",
    "explain",
    "what",
    "information",
}


def ask_model(live_config, question):
    generator = AIGenerator(live_config.OLLAMA_MODEL, live_config.OLLAMA_HOST)
    definition = CourseSearchTool(MagicMock()).get_tool_definition()
    response = generator.client.chat(
        model=generator.model,
        messages=[
            {"role": "system", "content": generator.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tools=generator._to_ollama_tools([definition]),
        options=generator.options,
    )
    return response.message


@pytest.mark.live
class TestLiveModel:
    @pytest.mark.parametrize("question", LIVE_QUESTIONS)
    def test_course_question_triggers_the_search_tool(self, live_config, question):
        message = ask_model(live_config, question)
        assert (
            message.tool_calls
        ), f"model answered without searching: {message.content[:120]!r}"
        assert message.tool_calls[0].function.name == "search_course_content"

    @pytest.mark.parametrize("question", LIVE_QUESTIONS)
    def test_search_query_text_is_meaningful(self, live_config, question):
        message = ask_model(live_config, question)
        if not message.tool_calls:
            pytest.skip("no tool call (covered by the test above)")
        query = (
            str(message.tool_calls[0].function.arguments.get("query", ""))
            .strip()
            .lower()
        )
        assert (
            query and query not in GENERIC_QUERIES
        ), f"query text too generic to retrieve anything: {query!r}"

    def test_general_knowledge_question_does_not_search(self, live_config):
        message = ask_model(live_config, "What is 2 + 2?")
        assert not message.tool_calls
