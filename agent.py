"""
Agentic orchestrator for UK Visa Assistant
Handles autonomous tool selection, execution, and iterative refinement
"""
import json
import traceback
from typing import List, Dict, Any
from datetime import datetime

import google.generativeai as genai

from models import AgentMemory, UserProfile
from tools import ToolImplementations
from tool_definitions import get_tools
from vector_db import EnhancedVectorDatabase
from config import GEMINI_MODEL_NAME


class AgenticVisaAssistant:
    """
    Fully agentic assistant using Gemini with function calling and search grounding
    Makes autonomous decisions about which tools to use
    """

    SYSTEM_INSTRUCTION = """You are an autonomous AI agent specialized in UK Skilled Worker visa sponsorship and job search.

You have access to these tools:
1. search_visa_sponsors - Find companies from UK government sponsor register
2. search_web_jobs - Find current job openings at companies
3. search_latest_company_news - Get LATEST real-time news and information using Google Search
4. match_jobs_to_resume - Match jobs to user's skills (requires resume)
5. check_visa_eligibility - Assess visa eligibility
6. analyze_resume - Parse resume file to extract profile
7. get_database_stats - Get system statistics

AUTONOMOUS DECISION MAKING:
- Analyze user query deeply
- Decide which tools to call and in what order
- Make multiple tool calls if needed to gather complete information
- When user asks for "latest", "recent", "current" info, ALWAYS use search_latest_company_news
- Reflect on results and decide if you need more information
- If results are insufficient, try different searches
- Maximum 5 tool calls per user query

STRATEGY:
1. For company-specific queries: search_visa_sponsors → search_web_jobs → search_latest_company_news (if asking about recent info)
2. For domain queries: search_visa_sponsors (broad) → filter best matches → search jobs
3. For resume-based queries: analyze_resume → match_jobs_to_resume
4. For eligibility questions: check_visa_eligibility (no other tools needed usually)
5. For latest/current company info: search_latest_company_news (uses Google Search grounding)

IMPORTANT:
- Always prioritize answering the user's specific question
- Use tools strategically - don't call tools unnecessarily
- If resume would help but isn't uploaded, suggest it
- Provide actionable, specific guidance
- Include direct links when available
- When using search_latest_company_news, summarize the findings naturally"""

    def __init__(self, gemini_api_key: str, vector_db: EnhancedVectorDatabase):
        """
        Initialize the agentic assistant

        Args:
            gemini_api_key: Google Gemini API key
            vector_db: Vector database instance for sponsor search
        """
        genai.configure(api_key=gemini_api_key)

        self.model_name = GEMINI_MODEL_NAME

        # Create model with proper tool configuration
        self.model = genai.GenerativeModel(
            self.model_name,
            system_instruction=self.SYSTEM_INSTRUCTION,
            tools=[get_tools()]
        )

        self.tools_impl = ToolImplementations(self.model, vector_db)
        self.memory = AgentMemory()

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return results

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Dictionary containing tool execution results
        """
        # Special handling for resume matching
        if tool_name == "match_jobs_to_resume":
            arguments['user_profile'] = self.memory.user_profile

        method = getattr(self.tools_impl, tool_name, None)
        if not method:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            result = method(**arguments)
        except Exception as e:
            result = {"success": False, "error": str(e)}

        # Save to history
        self.memory.tool_call_history.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

        return result

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """
        Safely serialize tool result for Gemini

        Args:
            result: Tool execution result

        Returns:
            Serialized result safe for Gemini consumption
        """
        try:
            # Convert UserProfile to dict if present
            if isinstance(result, dict):
                serialized = {}
                for key, value in result.items():
                    if isinstance(value, UserProfile):
                        serialized[key] = value.to_dict()
                    elif hasattr(value, '__dict__'):
                        serialized[key] = str(value)
                    else:
                        serialized[key] = value
                return serialized
            return result
        except Exception:
            return {"success": True, "data": str(result)}

    def chat(self, user_input: str, max_iterations: int = 5) -> str:
        """
        Main agentic loop with autonomous tool calling using Gemini

        Args:
            user_input: User's query/message
            max_iterations: Maximum number of tool-calling iterations

        Returns:
            Final response string from the agent
        """
        print(f"\n{'='*80}")
        print(f"👤 User: {user_input}")
        print(f"{'='*80}\n")

        # Add to conversation history
        self.memory.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Build conversation history for Gemini
        chat_history = []
        for msg in self.memory.conversation_history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # Add context about user profile
        context_parts = [user_input]
        if self.memory.user_profile:
            context_parts.append(f"\n[Context: User profile available with {len(self.memory.user_profile.skills)} skills]")

        context_msg = "".join(context_parts)

        tool_results_accumulated = []

        # Start chat session with history
        chat = self.model.start_chat(history=chat_history)

        try:
            # Initial message sending
            response = chat.send_message(context_msg)

            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                print(f"🔄 Agent Iteration {iteration}/{max_iterations}")

                # Check for function calls
                function_calls = self._extract_function_calls(response)

                if not function_calls:
                    # No function calls - we have the final response
                    break

                print(f"   🤖 Agent decided to call {len(function_calls)} tool(s)")

                # Execute function calls
                function_responses = []

                for fc in function_calls:
                    tool_name = fc.get('name')
                    tool_args = fc.get('args', {})

                    if not tool_name:
                        print(f"   ⚠️ Warning: Function call without name, skipping")
                        continue

                    print(f"   🔧 Calling: {tool_name}")
                    if tool_args:
                        print(f"      Args: {json.dumps(tool_args, indent=2)}")

                    # Execute the tool
                    result = self._execute_tool(tool_name, tool_args)

                    # Special handling for analyze_resume
                    if tool_name == "analyze_resume" and result.get('success'):
                        self.memory.user_profile = result.get('profile')
                        print(f"   ✅ Resume analyzed: {len(self.memory.user_profile.skills if self.memory.user_profile else [])} skills found")
                    else:
                        print(f"   ✅ Result: success={result.get('success', 'unknown')}")

                    tool_results_accumulated.append({
                        "tool": tool_name,
                        "result": result
                    })

                    # Serialize result for Gemini
                    serialized_result = self._serialize_result(result)

                    # Create proper function response
                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"result": serialized_result}
                            )
                        )
                    )

                # Send function responses back to model
                if function_responses:
                    try:
                        response = chat.send_message(function_responses)
                    except Exception as e:
                        print(f"   ⚠️ Error sending function response: {e}")
                        # Try sending as text fallback
                        fallback_text = json.dumps([
                            {"tool": tr["tool"], "success": tr["result"].get("success", False)}
                            for tr in tool_results_accumulated[-len(function_responses):]
                        ])
                        response = chat.send_message(f"Tool results: {fallback_text}")
                else:
                    # No valid function calls processed, break the loop
                    break

            # Extract final response text
            final_response = self._extract_text_response(response)

            print(f"\n{'='*80}")
            print("✅ Agent completed analysis")
            print(f"   Tools called: {len(tool_results_accumulated)}")
            print(f"   Iterations: {iteration}")
            print(f"{'='*80}\n")

            # Add to history
            self.memory.conversation_history.append({
                "role": "assistant",
                "content": final_response
            })

            return final_response

        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try again."
            print(f"❌ Error: {e}")
            traceback.print_exc()
            return error_msg

    def _extract_function_calls(self, response) -> List[Dict[str, Any]]:
        """Extract function calls from Gemini response"""
        function_calls = []

        try:
            if not hasattr(response, 'candidates') or len(response.candidates) == 0:
                return []

            candidate = response.candidates[0]

            if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                return []

            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call

                    # Extract name
                    name = None
                    if hasattr(fc, 'name'):
                        name = fc.name

                    # Extract args
                    args = {}
                    if hasattr(fc, 'args'):
                        # Convert MapComposite or similar to dict
                        try:
                            args = dict(fc.args)
                        except (TypeError, ValueError):
                            args = {}

                    if name:
                        function_calls.append({
                            'name': name,
                            'args': args
                        })
        except Exception as e:
            print(f"   ⚠️ Error extracting function calls: {e}")

        return function_calls

    def _extract_text_response(self, response) -> str:
        """Extract text from Gemini response"""
        try:
            if hasattr(response, 'text'):
                return response.text

            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "\n".join(text_parts)

            return "I processed your request but couldn't generate a response. Please try again."
        except Exception as e:
            return f"Error extracting response: {str(e)}"

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        base_stats = self.tools_impl.get_database_stats()
        stats = {
            "conversation_turns": len(self.memory.conversation_history) // 2,
            "tool_calls_made": len(self.memory.tool_call_history),
            "resume_uploaded": self.memory.user_profile is not None,
            "model": self.model_name
        }

        if base_stats.get('success'):
            stats.update({
                "total_sponsors": base_stats.get('total_sponsors', 0),
                "a_rated": base_stats.get('a_rated', 0),
                "unique_cities": base_stats.get('unique_cities', 0)
            })

        return stats
