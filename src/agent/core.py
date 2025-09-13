from typing import Dict, Any, List, Optional, Callable, TypedDict
from enum import Enum
import asyncio
from datetime import datetime
import uuid
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import json


class ExecutionPhase(Enum):
    """Execution phases for the agent."""
    IDLE = "idle"
    PLANNING = "planning" 
    TOOL_SELECTION = "tool_selection"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"


class ToolPermission(Enum):
    """Tool permission levels."""
    AUTO_APPROVE = "auto_approve"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCKED = "blocked"


@dataclass
class AgentState:
    """State object for the agent workflow."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_phase: ExecutionPhase = ExecutionPhase.IDLE
    plan: Optional[str] = None
    selected_tools: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    pending_approvals: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = None
    iteration_count: int = 0
    error_message: Optional[str] = None
    reflection: Optional[str] = None
    memory_context: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolDefinition:
    """Definition of a tool that can be used by the agent."""
    name: str
    description: str
    function: Callable
    parameters_schema: Dict[str, Any]
    permission: ToolPermission = ToolPermission.REQUIRE_CONFIRMATION
    category: str = "general"
    risk_level: int = 1  # 1-5, where 5 is highest risk


class TTTAgent:
    """Multi-step agent with planning, execution, and memory."""
    
    def __init__(self, llm_client, memory_manager, tool_registry):
        self.llm = llm_client
        self.memory = memory_manager
        self.tool_registry = tool_registry
        self.state = AgentState()
        self.workflow = self._build_workflow()
        self.phase_callbacks: Dict[ExecutionPhase, List[Callable]] = {
            phase: [] for phase in ExecutionPhase
        }
        
    def add_phase_callback(self, phase: ExecutionPhase, callback: Callable):
        """Add a callback for phase changes."""
        self.phase_callbacks[phase].append(callback)
    
    async def _notify_phase_change(self, new_phase: ExecutionPhase):
        """Notify all callbacks about phase change."""
        self.state.current_phase = new_phase
        for callback in self.phase_callbacks[new_phase]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(new_phase, self.state)
                else:
                    callback(new_phase, self.state)
            except Exception as e:
                print(f"Error in phase callback: {e}")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planning", self._planning_node)
        workflow.add_node("tool_selection", self._tool_selection_node)
        workflow.add_node("approval_check", self._approval_check_node)
        workflow.add_node("execution", self._execution_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("error_handling", self._error_handling_node)
        
        # Set entry point
        workflow.set_entry_point("planning")
        
        # Add edges
        workflow.add_edge("planning", "tool_selection")
        workflow.add_conditional_edges(
            "tool_selection",
            self._should_continue_to_approval,
            {
                "approval": "approval_check",
                "execute": "execution",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "approval_check", 
            self._approval_decision,
            {
                "approved": "execution",
                "rejected": "planning",
                "waiting": "approval_check"
            }
        )
        workflow.add_conditional_edges(
            "execution",
            self._execution_decision,
            {
                "continue": "reflection",
                "error": "error_handling",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "reflection",
            self._should_continue,
            {
                "continue": "planning", 
                "end": END
            }
        )
        workflow.add_edge("error_handling", END)
        
        return workflow.compile()
    
    async def _planning_node(self, state: AgentState) -> AgentState:
        """Planning phase - analyze request and create execution plan."""
        await self._notify_phase_change(ExecutionPhase.PLANNING)
        
        try:
            # Get relevant memory context
            if state.messages:
                last_message = state.messages[-1]["content"]
                memory_context = await self.memory.retrieve_relevant_context(
                    last_message, state.session_id, state.task_id
                )
                state.memory_context = memory_context
            
            # Create planning prompt
            planning_prompt = self._create_planning_prompt(state)
            
            # Get plan from LLM
            plan_response = await self.llm.generate_plan(planning_prompt)
            state.plan = plan_response.get("plan")
            state.selected_tools = plan_response.get("tools", [])
            
            return state
            
        except Exception as e:
            state.error_message = f"Planning error: {str(e)}"
            state.current_phase = ExecutionPhase.ERROR
            return state
    
    async def _tool_selection_node(self, state: AgentState) -> AgentState:
        """Tool selection and validation."""
        await self._notify_phase_change(ExecutionPhase.TOOL_SELECTION)
        
        # Validate selected tools exist and are available
        valid_tools = []
        for tool_name in state.selected_tools:
            if self.tool_registry.has_tool(tool_name):
                valid_tools.append(tool_name)
        
        state.selected_tools = valid_tools
        return state
    
    async def _approval_check_node(self, state: AgentState) -> AgentState:
        """Check if tools need approval."""
        await self._notify_phase_change(ExecutionPhase.AWAITING_APPROVAL)
        
        pending_approvals = []
        for tool_name in state.selected_tools:
            tool_def = self.tool_registry.get_tool(tool_name)
            if tool_def.permission == ToolPermission.REQUIRE_CONFIRMATION:
                pending_approvals.append(tool_name)
        
        state.pending_approvals = pending_approvals
        return state
    
    async def _execution_node(self, state: AgentState) -> AgentState:
        """Execute approved tools."""
        await self._notify_phase_change(ExecutionPhase.EXECUTING)
        
        try:
            for tool_name in state.selected_tools:
                if tool_name not in state.pending_approvals:
                    tool_def = self.tool_registry.get_tool(tool_name)
                    # Execute tool (simplified - would need proper parameter handling)
                    result = await self._execute_tool(tool_def, state)
                    state.tool_results[tool_name] = result
            
            return state
            
        except Exception as e:
            state.error_message = f"Execution error: {str(e)}"
            state.current_phase = ExecutionPhase.ERROR
            return state
    
    async def _reflection_node(self, state: AgentState) -> AgentState:
        """Reflect on results and decide next steps."""
        await self._notify_phase_change(ExecutionPhase.REFLECTING)
        
        # Analyze results and determine if task is complete
        reflection_prompt = self._create_reflection_prompt(state)
        reflection_response = await self.llm.reflect_on_results(reflection_prompt)
        
        state.reflection = reflection_response.get("reflection")
        state.iteration_count += 1
        
        # Store results in memory
        await self.memory.store_interaction(
            state.session_id,
            state.task_id,
            state.messages[-1] if state.messages else {},
            {
                "plan": state.plan,
                "tools_used": list(state.tool_results.keys()),
                "results": state.tool_results,
                "reflection": state.reflection
            }
        )
        
        return state
    
    async def _error_handling_node(self, state: AgentState) -> AgentState:
        """Handle errors gracefully."""
        await self._notify_phase_change(ExecutionPhase.ERROR)
        
        # Log error and attempt recovery
        error_context = {
            "error": state.error_message,
            "phase": state.current_phase.value,
            "iteration": state.iteration_count
        }
        
        # Store error in memory for learning
        await self.memory.store_error(state.session_id, error_context)
        
        return state
    
    def _should_continue_to_approval(self, state: AgentState) -> str:
        """Decide if we need approval or can execute directly."""
        if not state.selected_tools:
            return "end"
        
        needs_approval = any(
            self.tool_registry.get_tool(tool).permission == ToolPermission.REQUIRE_CONFIRMATION
            for tool in state.selected_tools
            if self.tool_registry.has_tool(tool)
        )
        
        return "approval" if needs_approval else "execute"
    
    def _approval_decision(self, state: AgentState) -> str:
        """Check approval status."""
        if not state.pending_approvals:
            return "approved"
        # In real implementation, this would check external approval status
        return "waiting"
    
    def _execution_decision(self, state: AgentState) -> str:
        """Decide next step after execution."""
        if state.error_message:
            return "error"
        return "continue"
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide if we should continue iterating."""
        from config.settings import settings
        
        if state.iteration_count >= settings.max_iterations:
            return "end"
        
        # Check if reflection indicates completion
        if state.reflection and "complete" in state.reflection.lower():
            return "end"
        
        return "continue"
    
    def _create_planning_prompt(self, state: AgentState) -> str:
        """Create prompt for planning phase."""
        context = ""
        if state.memory_context:
            context = "Previous relevant interactions:\n"
            for ctx in state.memory_context:
                context += f"- {ctx.get('summary', '')}\n"
        
        current_request = state.messages[-1]["content"] if state.messages else ""
        
        return f"""
        {context}
        
        Current request: {current_request}
        
        Available tools: {list(self.tool_registry.list_tools())}
        
        Create a step-by-step plan to fulfill this request. 
        Select appropriate tools and provide reasoning.
        
        Respond with JSON:
        {{
            "plan": "detailed step-by-step plan",
            "tools": ["tool1", "tool2"],
            "reasoning": "why these tools were selected"
        }}
        """
    
    def _create_reflection_prompt(self, state: AgentState) -> str:
        """Create prompt for reflection phase."""
        return f"""
        Plan executed: {state.plan}
        Tools used: {list(state.tool_results.keys())}
        Results: {state.tool_results}
        
        Analyze the results and determine:
        1. Was the task completed successfully?
        2. What was learned?
        3. Are additional steps needed?
        
        Respond with JSON:
        {{
            "reflection": "analysis of results",
            "completed": true/false,
            "next_steps": "what to do next if not completed"
        }}
        """
    
    async def _execute_tool(self, tool_def: ToolDefinition, state: AgentState) -> Any:
        """Execute a specific tool."""
        # Simplified - in real implementation would parse parameters from plan
        try:
            result = await tool_def.function()
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def process_message(self, message: str, session_id: str = None, task_id: str = None) -> AgentState:
        """Process a new message through the workflow."""
        if session_id:
            self.state.session_id = session_id
        if task_id:
            self.state.task_id = task_id
        
        # Add message to state
        self.state.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Run workflow
        final_state = await self.workflow.ainvoke(self.state)
        
        await self._notify_phase_change(ExecutionPhase.COMPLETED)
        return final_state
    
    async def approve_tools(self, approved_tools: List[str]):
        """Approve specific tools for execution."""
        self.state.pending_approvals = [
            tool for tool in self.state.pending_approvals 
            if tool not in approved_tools
        ]
    
    async def reject_tools(self, rejected_tools: List[str]):
        """Reject specific tools."""
        self.state.selected_tools = [
            tool for tool in self.state.selected_tools
            if tool not in rejected_tools
        ]
