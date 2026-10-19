import ollama
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with a local Ollama model for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to 2 sequential tool calls per query**, each one after you have seen the previous result
- Make a second call only when it depends on the first result or the first result is insufficient (e.g. get a course outline, then search for the lesson topic it reveals)
- Never repeat an identical call; after 2 rounds, answer from what you have
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Sequential tool rounds allowed per user query
    MAX_TOOL_ROUNDS = 2

    def __init__(self, model: str, host: str):
        self.client = ollama.Client(host=host)
        self.model = model

        # Pre-build base chat options
        self.options = {
            "temperature": 0,
            "num_predict": 800,
            "num_ctx": 8192
        }

    @staticmethod
    def _to_ollama_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool definitions (input_schema style) to Ollama's function format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        chat_params = {
            "model": self.model,
            "messages": messages,
            "options": self.options,
        }
        if tools:
            chat_params["tools"] = self._to_ollama_tools(tools)

        # Each round is its own request; the model sees earlier results via `messages`
        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.client.chat(**chat_params)

            if not response.message.tool_calls or not tool_manager:
                return response.message.content

            messages.append(response.message)
            if not self._execute_tool_calls(response.message.tool_calls, messages, tool_manager):
                break

        # Rounds used up (or a tool failed): final call without tools so the model has to answer
        chat_params.pop("tools", None)
        return self.client.chat(**chat_params).message.content

    def _execute_tool_calls(self, tool_calls, messages: List[Dict[str, Any]], tool_manager) -> bool:
        """
        Run every tool call from one response, appending a tool message per call.

        Returns:
            False if any call raised (the error text is sent to the model), else True
        """
        succeeded = True
        for call in tool_calls:
            try:
                tool_result = tool_manager.execute_tool(
                    call.function.name,
                    **self._clean_arguments(call.function.arguments)
                )
            except Exception as e:
                tool_result = f"Tool error: {e}"
                succeeded = False

            messages.append({
                "role": "tool",
                "tool_name": call.function.name,
                "content": tool_result,
            })
        return succeeded

    @staticmethod
    def _clean_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Drop empty values and coerce lesson_number, since small models often send strings"""
        cleaned = {k: v for k, v in dict(arguments).items() if v not in (None, "", "null")}
        if "lesson_number" in cleaned:
            try:
                cleaned["lesson_number"] = int(cleaned["lesson_number"])
            except (TypeError, ValueError):
                del cleaned["lesson_number"]
        return cleaned
