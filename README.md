# TTT-Agent 🤖

**Agentic Chat System with Human Confirmation and Transparent Execution**

An intelligent multi-step agent system built with LangGraph, ChromaDB, and Gradio that provides transparent execution with human oversight for tool usage.

## ✨ Features

- **🤖 Multi-Step Agent**: Powered by LangGraph with planning and execution phases
- **🔍 Transparent Execution**: Real-time visibility into agent planning and tool execution
- **✋ Human Confirmation**: Configurable approval system for tool usage with auto-approval flags
- **🧠 Vector Memory**: ChromaDB for conversation history and task-specific memory
- **🔧 Tool Registry**: Extensible tool system with permission management
- **💬 Chat Interface**: Gradio-based UI with execution phase indicators
- **🎯 Gemini 2.5 Pro**: Advanced language model for planning and reasoning
- **📊 Memory Management**: Conversation history + task-specific memory with semantic search

## 🏗️ Architecture

```
TTT-Agent/
├── src/
│   ├── agent/           # Core agent logic
│   │   ├── core.py      # LangGraph workflow & state management
│   │   └── gemini_client.py  # Gemini API integration
│   ├── tools/           # Tool system
│   │   ├── registry.py  # Tool registration & permission system
│   │   └── default_tools.py  # Built-in tools
│   ├── memory/          # Memory management
│   │   └── manager.py   # ChromaDB integration
│   ├── ui/              # User interface
│   │   └── gradio_interface.py  # Gradio chat UI
│   └── main.py          # Application entry point
├── config/              # Configuration
│   └── settings.py      # Environment settings
├── data/                # Data storage
│   ├── chroma_db/       # Vector database
│   └── logs/            # Application logs
└── scripts/             # Utility scripts
    ├── setup.sh         # Environment setup
    └── run.sh           # Run application
```
