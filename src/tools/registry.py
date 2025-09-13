from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import asyncio
import inspect
from src.agent.core import ToolDefinition, ToolPermission


class ToolRegistry:
    """Registry for managing tools and their permissions."""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters_schema: Dict[str, Any],
        permission: ToolPermission = ToolPermission.REQUIRE_CONFIRMATION,
        category: str = "general",
        risk_level: int = 1
    ) -> None:
        """Register a new tool."""
        tool_def = ToolDefinition(
            name=name,
            description=description,
            function=function,
            parameters_schema=parameters_schema,
            permission=permission,
            category=category,
            risk_level=risk_level
        )
        
        self._tools[name] = tool_def
        
        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            tool_def = self._tools[name]
            del self._tools[name]
            
            # Update category index
            if tool_def.category in self._categories:
                if name in self._categories[tool_def.category]:
                    self._categories[tool_def.category].remove(name)
            
            return True
        return False
    
    def has_tool(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools in a specific category."""
        return self._categories.get(category, [])
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self._categories.keys())
    
    def set_tool_permission(self, name: str, permission: ToolPermission) -> bool:
        """Update tool permission."""
        if name in self._tools:
            self._tools[name].permission = permission
            return True
        return False
    
    def get_safe_tools(self) -> List[str]:
        """Get tools marked for auto-approval."""
        return [
            name for name, tool_def in self._tools.items()
            if tool_def.permission == ToolPermission.AUTO_APPROVE
        ]
    
    def get_tools_requiring_approval(self) -> List[str]:
        """Get tools requiring human confirmation."""
        return [
            name for name, tool_def in self._tools.items()
            if tool_def.permission == ToolPermission.REQUIRE_CONFIRMATION
        ]
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive tool information."""
        if name not in self._tools:
            return {}
        
        tool_def = self._tools[name]
        return {
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters_schema": tool_def.parameters_schema,
            "permission": tool_def.permission.value,
            "category": tool_def.category,
            "risk_level": tool_def.risk_level,
            "is_async": asyncio.iscoroutinefunction(tool_def.function)
        }
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Export all tools as dictionary."""
        return {
            name: self.get_tool_info(name)
            for name in self._tools.keys()
        }


def tool_decorator(
    name: str = None,
    description: str = "",
    permission: ToolPermission = ToolPermission.REQUIRE_CONFIRMATION,
    category: str = "general", 
    risk_level: int = 1
):
    """Decorator to register tools."""
    def decorator(func: Callable):
        tool_name = name or func.__name__
        
        # Extract parameters schema from function signature
        sig = inspect.signature(func)
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
                elif param.annotation == list:
                    param_info["type"] = "array"
                elif param.annotation == dict:
                    param_info["type"] = "object"
            
            parameters_schema["properties"][param_name] = param_info
            
            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                parameters_schema["required"].append(param_name)
        
        # Store registration info on function
        func._tool_registry_info = {
            "name": tool_name,
            "description": description,
            "parameters_schema": parameters_schema,
            "permission": permission,
            "category": category,
            "risk_level": risk_level
        }
        
        return func
    return decorator


# Global tool registry instance
global_tool_registry = ToolRegistry()


def register_with_global_registry(func: Callable) -> None:
    """Register a function with the global tool registry."""
    if hasattr(func, '_tool_registry_info'):
        info = func._tool_registry_info
        global_tool_registry.register_tool(
            name=info["name"],
            function=func,
            description=info["description"],
            parameters_schema=info["parameters_schema"],
            permission=info["permission"],
            category=info["category"],
            risk_level=info["risk_level"]
        )
