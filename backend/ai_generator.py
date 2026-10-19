import ollama
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with a local Ollama model for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
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

        response = self.client.chat(**chat_params)

        # Handle tool execution if the model requested it
        if response.message.tool_calls and tool_manager:
            return self._handle_tool_execution(response, messages, tool_manager)

        return response.message.content

    def _handle_tool_execution(self, initial_response, messages: List[Dict[str, Any]], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool call requests
            messages: Conversation so far (system + user)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = messages.copy()
        messages.append(initial_response.message)

        for call in initial_response.message.tool_calls:
            try:
                tool_result = tool_manager.execute_tool(
                    call.function.name,
                    **self._clean_arguments(call.function.arguments)
                )
            except Exception as e:
                tool_result = f"Tool error: {e}"

            messages.append({
                "role": "tool",
                "tool_name": call.function.name,
                "content": tool_result,
            })

        # Final call without tools so the model has to answer
        final_response = self.client.chat(
            model=self.model,
            messages=messages,
            options=self.options,
        )
        return final_response.message.content

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
