import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import gradio as gr
import threading
import queue

from src.agent.core import TTTAgent, ExecutionPhase, AgentState
from src.agent.gemini_client import GeminiClient
from src.memory.manager import MemoryManager
from src.tools.registry import global_tool_registry
from src.tools.default_tools import register_default_tools
from config.settings import settings


class TTTAgentUI:
    """Gradio-based chat interface for TTT-Agent."""
    
    def __init__(self):
        self.agent = None
        self.memory_manager = None
        self.llm_client = None
        self.current_session_id = str(uuid.uuid4())
        self.current_task_id = None
        self.pending_approvals = []
        self.execution_status = "Ready"
        self.chat_history = []
        self.message_queue = queue.Queue()
        
        # UI state
        self.phase_display = "🟢 Ready"
        self.tool_approval_visible = False
        self.pending_tools_display = ""
        
    async def initialize(self):
        """Initialize all components."""
        try:
            print("🚀 Initializing TTT-Agent...")
            
            # Initialize memory manager
            self.memory_manager = MemoryManager(settings.chroma_db_path)
            await self.memory_manager.initialize()
            
            # Initialize LLM client
            self.llm_client = GeminiClient()
            
            # Register default tools
            register_default_tools()
            
            # Initialize agent
            self.agent = TTTAgent(
                llm_client=self.llm_client,
                memory_manager=self.memory_manager,
                tool_registry=global_tool_registry
            )
            
            # Setup phase callbacks
            self.agent.add_phase_callback(ExecutionPhase.PLANNING, self._on_planning_phase)
            self.agent.add_phase_callback(ExecutionPhase.AWAITING_APPROVAL, self._on_approval_phase)
            self.agent.add_phase_callback(ExecutionPhase.EXECUTING, self._on_executing_phase)
            self.agent.add_phase_callback(ExecutionPhase.COMPLETED, self._on_completed_phase)
            self.agent.add_phase_callback(ExecutionPhase.ERROR, self._on_error_phase)
            
            print("✅ TTT-Agent initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize TTT-Agent: {e}")
            return False
    
    async def _on_planning_phase(self, phase: ExecutionPhase, state: AgentState):
        """Handle planning phase updates."""
        self.phase_display = "🟡 Planning..."
        self.execution_status = "Analyzing request and creating execution plan"
    
    async def _on_approval_phase(self, phase: ExecutionPhase, state: AgentState):
        """Handle approval phase updates."""
        self.phase_display = "🔶 Awaiting Approval"
        self.execution_status = "Waiting for tool approval"
        self.pending_approvals = state.pending_approvals.copy()
        self.tool_approval_visible = len(self.pending_approvals) > 0
        
        if self.pending_approvals:
            tools_info = []
            for tool_name in self.pending_approvals:
                tool_info = global_tool_registry.get_tool_info(tool_name)
                tools_info.append(f"• **{tool_name}**: {tool_info.get('description', 'No description')}")
            self.pending_tools_display = "\n".join(tools_info)
    
    async def _on_executing_phase(self, phase: ExecutionPhase, state: AgentState):
        """Handle execution phase updates."""
        self.phase_display = "🔵 Executing..."
        self.execution_status = f"Executing {len(state.selected_tools)} tools"
        self.tool_approval_visible = False
    
    async def _on_completed_phase(self, phase: ExecutionPhase, state: AgentState):
        """Handle completion phase updates."""
        self.phase_display = "🟢 Completed"
        self.execution_status = "Task completed"
        self.tool_approval_visible = False
        self.pending_approvals = []
    
    async def _on_error_phase(self, phase: ExecutionPhase, state: AgentState):
        """Handle error phase updates."""
        self.phase_display = "🔴 Error"
        self.execution_status = f"Error: {state.error_message}"
        self.tool_approval_visible = False
    
    async def process_message(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]], str, bool, str]:
        """Process user message and return updated interface state."""
        if not message.strip():
            return "", history, self.phase_display, self.tool_approval_visible, self.pending_tools_display
        
        try:
            # Add user message to history
            history.append([message, None])
            
            # Store conversation in memory
            await self.memory_manager.store_conversation(
                session_id=self.current_session_id,
                role="user",
                content=message
            )
            
            # Create new task ID for this interaction
            self.current_task_id = str(uuid.uuid4())
            
            # Process message through agent
            final_state = await self.agent.process_message(
                message=message,
                session_id=self.current_session_id,
                task_id=self.current_task_id
            )
            
            # Generate response based on final state
            if final_state.error_message:
                response = f"I encountered an error: {final_state.error_message}"
            else:
                # Create a summary response
                response_parts = []
                
                if final_state.plan:
                    response_parts.append(f"**Plan**: {final_state.plan}")
                
                if final_state.tool_results:
                    response_parts.append("**Results**:")
                    for tool_name, result in final_state.tool_results.items():
                        if result.get("success"):
                            response_parts.append(f"✅ {tool_name}: {result.get('result', 'Completed')}")
                        else:
                            response_parts.append(f"❌ {tool_name}: {result.get('error', 'Failed')}")
                
                if final_state.reflection:
                    response_parts.append(f"**Reflection**: {final_state.reflection}")
                
                response = "\n\n".join(response_parts) if response_parts else "Task processed successfully."
            
            # Update history with response
            history[-1][1] = response
            
            # Store agent response in memory
            await self.memory_manager.store_conversation(
                session_id=self.current_session_id,
                role="assistant",
                content=response
            )
            
            return "", history, self.phase_display, self.tool_approval_visible, self.pending_tools_display
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            history[-1][1] = error_msg
            return "", history, "🔴 Error", False, ""
    
    async def approve_tools(self, approved_tools: List[str]) -> Tuple[str, bool, str]:
        """Approve selected tools for execution."""
        if self.agent and self.pending_approvals:
            await self.agent.approve_tools(approved_tools)
            
            # Remove approved tools from pending list
            self.pending_approvals = [tool for tool in self.pending_approvals if tool not in approved_tools]
            
            if not self.pending_approvals:
                self.tool_approval_visible = False
                self.phase_display = "🔵 Executing..."
                self.execution_status = "Tools approved, executing..."
            
        return self.phase_display, self.tool_approval_visible, self.pending_tools_display
    
    async def reject_tools(self, rejected_tools: List[str]) -> Tuple[str, bool, str]:
        """Reject selected tools."""
        if self.agent and self.pending_approvals:
            await self.agent.reject_tools(rejected_tools)
            
            # Remove rejected tools from pending list
            self.pending_approvals = [tool for tool in self.pending_approvals if tool not in rejected_tools]
            
            if not self.pending_approvals:
                self.tool_approval_visible = False
                self.phase_display = "🟡 Replanning..."
                self.execution_status = "Tools rejected, creating new plan..."
            
        return self.phase_display, self.tool_approval_visible, self.pending_tools_display
    
    def get_memory_stats(self) -> str:
        """Get memory statistics display."""
        try:
            if self.memory_manager:
                stats = asyncio.run(self.memory_manager.get_memory_stats())
                return f"""
                **Memory Statistics**
                - Conversations: {stats.get('conversations', 0)}
                - Tasks: {stats.get('tasks', 0)}
                - Interactions: {stats.get('interactions', 0)}
                """
            return "Memory manager not initialized"
        except Exception as e:
            return f"Error retrieving stats: {e}"
    
    def get_tool_registry_info(self) -> str:
        """Get tool registry information."""
        try:
            tools = global_tool_registry.to_dict()
            tool_info = []
            
            for name, info in tools.items():
                permission = info['permission']
                risk = info['risk_level']
                category = info['category']
                
                emoji = "🟢" if permission == "auto_approve" else "🔶" if permission == "require_confirmation" else "🔴"
                tool_info.append(f"{emoji} **{name}** ({category}) - Risk: {risk}/5")
            
            return "\n".join(tool_info) if tool_info else "No tools registered"
        except Exception as e:
            return f"Error retrieving tools: {e}"
    
    def new_session(self) -> Tuple[List[List[str]], str]:
        """Start a new chat session."""
        self.current_session_id = str(uuid.uuid4())
        self.current_task_id = None
        self.pending_approvals = []
        self.execution_status = "Ready"
        self.phase_display = "🟢 Ready"
        self.tool_approval_visible = False
        self.pending_tools_display = ""
        
        return [], "New session started"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        def sync_process_message(message, history):
            """Synchronous wrapper for async message processing."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.process_message(message, history))
            finally:
                loop.close()
        
        def sync_approve_tools():
            """Approve all pending tools."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.approve_tools(self.pending_approvals.copy()))
            finally:
                loop.close()
        
        def sync_reject_tools():
            """Reject all pending tools."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.reject_tools(self.pending_approvals.copy()))
            finally:
                loop.close()
        
        with gr.Blocks(
            title="TTT-Agent",
            theme=gr.themes.Soft(),
            css="""
                .status-indicator { font-weight: bold; font-size: 1.1em; }
                .tool-approval { background-color: #fff3cd; padding: 15px; border-radius: 5px; }
                .phase-display { font-size: 1.2em; font-weight: bold; }
            """
        ) as interface:
            
            gr.Markdown("# 🤖 TTT-Agent")
            gr.Markdown("*Agentic Chat System with Human Confirmation and Transparent Execution*")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=400,
                        show_label=True,
                        container=True
                    )
                    
                    msg_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("New Session", variant="secondary")
                
                with gr.Column(scale=1):
                    # Status panel
                    gr.Markdown("## Status")
                    
                    phase_status = gr.Markdown(
                        value="🟢 Ready",
                        elem_classes=["phase-display"]
                    )
                    
                    execution_details = gr.Textbox(
                        label="Current Activity",
                        value="Ready for input",
                        interactive=False,
                        lines=2
                    )
            
            # Tool approval section (hidden by default)
            with gr.Row(visible=False) as approval_section:
                with gr.Column():
                    gr.Markdown("## 🔶 Tool Approval Required", elem_classes=["tool-approval"])
                    
                    pending_tools = gr.Markdown(
                        value="",
                        label="Pending Tools"
                    )
                    
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve All", variant="primary")
                        reject_btn = gr.Button("❌ Reject All", variant="secondary")
            
            # Information tabs
            with gr.Tabs():
                with gr.Tab("Memory Stats"):
                    memory_stats = gr.Markdown(value=self.get_memory_stats())
                    refresh_memory_btn = gr.Button("Refresh Stats")
                
                with gr.Tab("Available Tools"):
                    tool_list = gr.Markdown(value=self.get_tool_registry_info())
                
                with gr.Tab("Settings"):
                    gr.Markdown(f"""
                    **Current Configuration:**
                    - Session ID: `{self.current_session_id}`
                    - Agent: {settings.agent_name}
                    - Model: Gemini 2.5 Pro
                    - Memory: ChromaDB
                    - Auto-approve safe tools: {settings.auto_approve_safe_tools}
                    """)
            
            # Event handlers
            def handle_send(message, history):
                return sync_process_message(message, history)
            
            def handle_approve():
                phase, visible, tools = sync_approve_tools()
                return phase, gr.update(visible=visible), tools
            
            def handle_reject():
                phase, visible, tools = sync_reject_tools()
                return phase, gr.update(visible=visible), tools
            
            # Wire up events
            send_btn.click(
                handle_send,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, phase_status, approval_section, pending_tools]
            )
            
            msg_input.submit(
                handle_send,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot, phase_status, approval_section, pending_tools]
            )
            
            approve_btn.click(
                handle_approve,
                outputs=[phase_status, approval_section, pending_tools]
            )
            
            reject_btn.click(
                handle_reject,
                outputs=[phase_status, approval_section, pending_tools]
            )
            
            clear_btn.click(
                self.new_session,
                outputs=[chatbot, execution_details]
            )
            
            refresh_memory_btn.click(
                self.get_memory_stats,
                outputs=[memory_stats]
            )
        
        return interface


# Global UI instance
ui_instance = None


async def create_ui():
    """Create and initialize the UI."""
    global ui_instance
    
    ui_instance = TTTAgentUI()
    success = await ui_instance.initialize()
    
    if not success:
        raise Exception("Failed to initialize TTT-Agent")
    
    return ui_instance.create_interface()


def launch_ui():
    """Launch the Gradio interface."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        interface = loop.run_until_complete(create_ui())
        
        interface.launch(
            server_name=settings.gradio_host,
            server_port=settings.gradio_port,
            share=False,
            debug=True,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"❌ Failed to launch UI: {e}")
        raise
    finally:
        loop.close()


if __name__ == "__main__":
    launch_ui()
