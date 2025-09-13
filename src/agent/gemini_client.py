import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings


class GeminiClient:
    """Google Gemini 2.5 Pro API client with rate limiting and error handling."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        try:
            # Initialize the new Google Gen AI client
            self.client = genai.Client(api_key=self.api_key)
            
            print("✅ Gemini client initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini client: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, prompt: str, config: types.GenerateContentConfig = None) -> str:
        """Make a request to Gemini with retry logic."""
        try:
            response = await self.client.aio.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=prompt,
                config=config
            )
            
            if response.text:
                return response.text
            else:
                raise Exception("No response generated")
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise
    
    async def generate_plan(self, prompt: str) -> Dict[str, Any]:
        """Generate a plan using Gemini."""
        planning_system_prompt = """
        You are an intelligent planning agent. Your job is to analyze user requests and create detailed execution plans.
        
        When given a request:
        1. Break it down into clear, actionable steps
        2. Identify what tools would be needed for each step
        3. Consider dependencies between steps
        4. Provide reasoning for your approach
        
        Always respond with valid JSON in this format:
        {
            "plan": "Detailed step-by-step plan as a string",
            "tools": ["tool1", "tool2", "tool3"],
            "reasoning": "Why this approach and these tools",
            "estimated_complexity": "low|medium|high",
            "dependencies": ["any prerequisites or considerations"]
        }
        
        Available tool categories: filesystem, system, web, utility
        """
        
        full_prompt = f"{planning_system_prompt}\n\nUser Request: {prompt}"
        
        try:
            # Configure for structured JSON response
            config = types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type='application/json'
            )
            
            response_text = await self._make_request(full_prompt, config)
            
            # Try to parse JSON response
            try:
                plan_data = json.loads(response_text)
                
                # Validate required fields
                required_fields = ["plan", "tools", "reasoning"]
                for field in required_fields:
                    if field not in plan_data:
                        plan_data[field] = f"Generated {field}"
                
                return plan_data
                
            except json.JSONDecodeError:
                # Fallback: create structured response from text
                return {
                    "plan": response_text,
                    "tools": [],
                    "reasoning": "Fallback parsing - could not extract JSON",
                    "estimated_complexity": "unknown"
                }
                
        except Exception as e:
            return {
                "plan": f"Error generating plan: {str(e)}",
                "tools": [],
                "reasoning": "Error occurred during planning",
                "error": str(e)
            }
    
    async def reflect_on_results(self, prompt: str) -> Dict[str, Any]:
        """Reflect on execution results and determine next steps."""
        reflection_system_prompt = """
        You are a reflective agent that analyzes task execution results and determines next steps.
        
        When given execution results:
        1. Analyze what was accomplished
        2. Identify any issues or failures
        3. Determine if the original goal was met
        4. Suggest improvements or next steps
        5. Extract key learnings
        
        Always respond with valid JSON in this format:
        {
            "reflection": "Analysis of what happened and results",
            "completed": true/false,
            "success_rate": 0.0-1.0,
            "next_steps": "What should happen next (if not completed)",
            "learnings": "Key insights or lessons learned",
            "issues_found": ["any problems encountered"],
            "recommendations": ["suggestions for improvement"]
        }
        """
        
        full_prompt = f"{reflection_system_prompt}\n\nExecution Results: {prompt}"
        
        try:
            # Configure for structured JSON response
            config = types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type='application/json'
            )
            
            response_text = await self._make_request(full_prompt, config)
            
            # Try to parse JSON response
            try:
                reflection_data = json.loads(response_text)
                
                # Validate and set defaults
                if "reflection" not in reflection_data:
                    reflection_data["reflection"] = response_text
                if "completed" not in reflection_data:
                    reflection_data["completed"] = False
                    
                return reflection_data
                
            except json.JSONDecodeError:
                # Fallback: create structured response from text
                return {
                    "reflection": response_text,
                    "completed": False,
                    "success_rate": 0.5,
                    "next_steps": "Review and retry",
                    "learnings": "Could not parse structured reflection"
                }
                
        except Exception as e:
            return {
                "reflection": f"Error during reflection: {str(e)}",
                "completed": False,
                "success_rate": 0.0,
                "error": str(e)
            }
    
    async def generate_response(
        self,
        message: str,
        context: List[Dict[str, Any]] = None,
        system_prompt: str = None
    ) -> str:
        """Generate a conversational response."""
        default_system_prompt = """
        You are TTT-Agent, an intelligent assistant that helps users accomplish tasks through planning and tool execution.
        
        You are helpful, accurate, and transparent about your capabilities and limitations.
        When users ask questions or request tasks, you think through the problem systematically.
        
        Key traits:
        - Clear and concise communication
        - Honest about uncertainties
        - Proactive in suggesting solutions
        - Respectful of user preferences and safety
        """
        
        context_text = ""
        if context:
            context_text = "\nRelevant context from previous interactions:\n"
            for ctx in context[:3]:  # Limit context to avoid token overflow
                context_text += f"- {ctx.get('content', '')[:200]}...\n"
        
        full_prompt = f"""{system_prompt or default_system_prompt}
        
        {context_text}
        
        User: {message}
        
        Assistant:"""
        
        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
            
            response = await self._make_request(full_prompt, config)
            return response.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    async def start_chat_session(self) -> None:
        """Start a new chat session for conversation continuity."""
        try:
            # With the new SDK, we don't need a separate chat session
            # The client handles conversation context automatically
            print("✅ Chat session ready")
        except Exception as e:
            print(f"❌ Failed to start chat session: {e}")
            raise
    
    async def send_message(self, message: str) -> str:
        """Send a message in the ongoing chat session."""
        try:
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048
            )
            
            response = await self.client.aio.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=message,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Chat error: {e}")
            return f"Error in chat: {str(e)}"
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            # List available models
            models = []
            for model in self.client.models.list():
                models.append({
                    "name": model.name,
                    "display_name": getattr(model, 'display_name', model.name),
                    "description": getattr(model, 'description', 'No description available')
                })
            
            # Find our current model
            current_model_info = None
            for model in models:
                if "gemini-2.0-flash" in model["name"]:
                    current_model_info = model
                    break
            
            return current_model_info or {"error": "Model information not available", "available_models": models}
            
        except Exception as e:
            return {"error": f"Could not retrieve model info: {str(e)}"}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the Gemini connection."""
        try:
            config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=10)
            test_response = await self._make_request("Hello! This is a test message. Please respond with 'OK'.", config)
            
            return {
                "status": "healthy",
                "api_accessible": True,
                "test_response": test_response[:100] + "..." if len(test_response) > 100 else test_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "api_accessible": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
