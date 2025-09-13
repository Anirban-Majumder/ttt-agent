from typing import Dict, List, Any, Optional
import asyncio
import subprocess
import os
import json
import requests
from datetime import datetime
import random

from src.tools.registry import tool_decorator, ToolPermission, register_with_global_registry


# File System Tools
@tool_decorator(
    name="read_file",
    description="Read contents of a file",
    permission=ToolPermission.AUTO_APPROVE,
    category="filesystem",
    risk_level=1
)
async def read_file(file_path: str) -> Dict[str, Any]:
    """Read file contents safely."""
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": "File not found"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "content": content,
            "size": len(content),
            "path": file_path
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool_decorator(
    name="write_file", 
    description="Write content to a file",
    permission=ToolPermission.REQUIRE_CONFIRMATION,
    category="filesystem",
    risk_level=3
)
async def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """Write content to file with confirmation."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "message": f"File written successfully: {file_path}",
            "size": len(content)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool_decorator(
    name="list_directory",
    description="List files and directories in a path",
    permission=ToolPermission.AUTO_APPROVE,
    category="filesystem", 
    risk_level=1
)
async def list_directory(directory_path: str) -> Dict[str, Any]:
    """List directory contents."""
    try:
        if not os.path.exists(directory_path):
            return {"success": False, "error": "Directory not found"}
        
        items = []
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            items.append({
                "name": item,
                "type": "directory" if os.path.isdir(item_path) else "file",
                "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None,
                "modified": datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
            })
        
        return {
            "success": True,
            "path": directory_path,
            "items": items,
            "count": len(items)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# System Tools
@tool_decorator(
    name="run_command",
    description="Execute a system command",
    permission=ToolPermission.REQUIRE_CONFIRMATION,
    category="system",
    risk_level=4
)
async def run_command(command: str, timeout: int = 30) -> Dict[str, Any]:
    """Run system command with timeout."""
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), 
            timeout=timeout
        )
        
        return {
            "success": True,
            "command": command,
            "return_code": process.returncode,
            "stdout": stdout.decode('utf-8'),
            "stderr": stderr.decode('utf-8')
        }
    except asyncio.TimeoutError:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool_decorator(
    name="get_system_info",
    description="Get basic system information", 
    permission=ToolPermission.AUTO_APPROVE,
    category="system",
    risk_level=1
)
async def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    try:
        import platform
        import psutil
        
        return {
            "success": True,
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Web Tools
@tool_decorator(
    name="web_search",
    description="Search the web for information",
    permission=ToolPermission.REQUIRE_CONFIRMATION,
    category="web",
    risk_level=2
)
async def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Perform web search (mock implementation)."""
    # Mock implementation - in real version would use search API
    try:
        results = []
        for i in range(min(num_results, 5)):
            results.append({
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a mock search result for the query '{query}'. " +
                          f"Result {i+1} contains relevant information about the topic."
            })
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool_decorator(
    name="fetch_url",
    description="Fetch content from a URL",
    permission=ToolPermission.REQUIRE_CONFIRMATION,
    category="web",
    risk_level=2
)
async def fetch_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Fetch URL content."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        return {
            "success": True,
            "url": url,
            "status_code": response.status_code,
            "content": response.text,
            "headers": dict(response.headers),
            "size": len(response.content)
        }
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Utility Tools
@tool_decorator(
    name="calculate",
    description="Perform mathematical calculations",
    permission=ToolPermission.AUTO_APPROVE,
    category="utility",
    risk_level=1
)
async def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate mathematical expressions."""
    try:
        # Allow only safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return {"success": False, "error": "Invalid characters in expression"}
        
        result = eval(expression)
        return {
            "success": True,
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool_decorator(
    name="generate_random",
    description="Generate random numbers or strings",
    permission=ToolPermission.AUTO_APPROVE,
    category="utility",
    risk_level=1
)
async def generate_random(
    type: str = "number",
    min_val: int = 1,
    max_val: int = 100,
    length: int = 10
) -> Dict[str, Any]:
    """Generate random values."""
    try:
        if type == "number":
            result = random.randint(min_val, max_val)
        elif type == "float":
            result = random.uniform(min_val, max_val)
        elif type == "string":
            import string
            chars = string.ascii_letters + string.digits
            result = ''.join(random.choice(chars) for _ in range(length))
        else:
            return {"success": False, "error": "Invalid type. Use 'number', 'float', or 'string'"}
        
        return {
            "success": True,
            "type": type,
            "result": result,
            "parameters": {"min_val": min_val, "max_val": max_val, "length": length}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool_decorator(
    name="get_current_time",
    description="Get current date and time",
    permission=ToolPermission.AUTO_APPROVE,
    category="utility",
    risk_level=1
)
async def get_current_time(format: str = "iso") -> Dict[str, Any]:
    """Get current timestamp."""
    try:
        now = datetime.now()
        
        if format == "iso":
            result = now.isoformat()
        elif format == "unix":
            result = now.timestamp()
        elif format == "human":
            result = now.strftime("%Y-%m-%d %H:%M:%S")
        else:
            result = str(now)
        
        return {
            "success": True,
            "timestamp": result,
            "format": format,
            "timezone": str(now.astimezone().tzinfo)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Register all tools with global registry
def register_default_tools():
    """Register all default tools with the global registry."""
    tools = [
        read_file, write_file, list_directory,
        run_command, get_system_info,
        web_search, fetch_url,
        calculate, generate_random, get_current_time
    ]
    
    for tool in tools:
        register_with_global_registry(tool)


# Auto-register when module is imported
register_default_tools()
