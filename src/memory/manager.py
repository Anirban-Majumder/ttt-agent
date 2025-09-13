from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


@dataclass
class ConversationMemory:
    """Memory entry for conversations."""
    session_id: str
    message_id: str
    timestamp: datetime
    role: str  # user, assistant, system
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TaskMemory:
    """Memory entry for tasks and projects."""
    task_id: str
    session_id: str
    timestamp: datetime
    title: str
    description: str
    plan: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionMemory:
    """Memory entry for agent interactions."""
    interaction_id: str
    session_id: str
    task_id: Optional[str]
    timestamp: datetime
    user_input: str
    agent_plan: str
    tools_executed: List[str]
    results: Dict[str, Any]
    reflection: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """Manages conversation history and task-specific memory using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.conversation_collection = None
        self.task_collection = None
        self.interaction_collection = None
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
    async def initialize(self):
        """Initialize ChromaDB collections."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collections
            self.conversation_collection = self.client.get_or_create_collection(
                name="conversations",
                embedding_function=self.embedding_function,
                metadata={"description": "Conversation history and context"}
            )
            
            self.task_collection = self.client.get_or_create_collection(
                name="tasks", 
                embedding_function=self.embedding_function,
                metadata={"description": "Task and project memories"}
            )
            
            self.interaction_collection = self.client.get_or_create_collection(
                name="interactions",
                embedding_function=self.embedding_function,
                metadata={"description": "Agent interaction history"}
            )
            
            print("✅ Memory manager initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize memory manager: {e}")
            raise
    
    async def store_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store a conversation message."""
        if not self.conversation_collection:
            await self.initialize()
        
        message_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        conversation_data = {
            "session_id": session_id,
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat(),
            **(metadata or {})
        }
        
        # Store in ChromaDB
        self.conversation_collection.add(
            documents=[content],
            metadatas=[conversation_data],
            ids=[message_id]
        )
        
        return message_id
    
    async def store_task_memory(
        self,
        task_id: str,
        session_id: str,
        title: str,
        description: str,
        plan: str = None,
        tools_used: List[str] = None,
        results: Dict[str, Any] = None,
        status: str = "active",
        metadata: Dict[str, Any] = None
    ) -> None:
        """Store task-specific memory."""
        if not self.task_collection:
            await self.initialize()
        
        timestamp = datetime.now()
        
        task_data = {
            "task_id": task_id,
            "session_id": session_id,
            "title": title,
            "description": description,
            "plan": plan,
            "tools_used": json.dumps(tools_used or []),
            "results": json.dumps(results or {}),
            "status": status,
            "timestamp": timestamp.isoformat(),
            **(metadata or {})
        }
        
        # Create document from task information
        document = f"{title}\n{description}\n{plan or ''}"
        
        self.task_collection.upsert(
            documents=[document],
            metadatas=[task_data],
            ids=[task_id]
        )
    
    async def store_interaction(
        self,
        session_id: str,
        task_id: Optional[str],
        user_input: Dict[str, Any],
        agent_response: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store complete interaction memory."""
        if not self.interaction_collection:
            await self.initialize()
        
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        interaction_data = {
            "interaction_id": interaction_id,
            "session_id": session_id,
            "task_id": task_id,
            "user_input": json.dumps(user_input),
            "agent_plan": agent_response.get("plan", ""),
            "tools_executed": json.dumps(agent_response.get("tools_used", [])),
            "results": json.dumps(agent_response.get("results", {})),
            "reflection": agent_response.get("reflection"),
            "timestamp": timestamp.isoformat(),
            **(metadata or {})
        }
        
        # Create searchable document
        document = f"""
        User: {user_input.get('content', '')}
        Plan: {agent_response.get('plan', '')}
        Tools: {', '.join(agent_response.get('tools_used', []))}
        Reflection: {agent_response.get('reflection', '')}
        """
        
        self.interaction_collection.add(
            documents=[document.strip()],
            metadatas=[interaction_data],
            ids=[interaction_id]
        )
        
        return interaction_id
    
    async def retrieve_conversation_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[ConversationMemory]:
        """Retrieve recent conversation history for a session."""
        if not self.conversation_collection:
            await self.initialize()
        
        try:
            results = self.conversation_collection.get(
                where={"session_id": session_id},
                limit=limit
            )
            
            conversations = []
            for i, metadata in enumerate(results["metadatas"]):
                conversations.append(ConversationMemory(
                    session_id=metadata["session_id"],
                    message_id=metadata["message_id"],
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    role=metadata["role"],
                    content=results["documents"][i],
                    metadata=metadata
                ))
            
            # Sort by timestamp
            conversations.sort(key=lambda x: x.timestamp)
            return conversations
            
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    async def retrieve_relevant_context(
        self,
        query: str,
        session_id: str,
        task_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context using semantic search."""
        if not all([self.conversation_collection, self.task_collection, self.interaction_collection]):
            await self.initialize()
        
        context = []
        
        try:
            # Search conversations
            conv_results = self.conversation_collection.query(
                query_texts=[query],
                where={"session_id": session_id},
                n_results=min(limit, 3)
            )
            
            for i, doc in enumerate(conv_results["documents"][0]):
                metadata = conv_results["metadatas"][0][i]
                context.append({
                    "type": "conversation",
                    "content": doc,
                    "timestamp": metadata["timestamp"],
                    "role": metadata["role"],
                    "distance": conv_results["distances"][0][i]
                })
            
            # Search tasks if task_id provided
            if task_id:
                task_results = self.task_collection.query(
                    query_texts=[query],
                    where={"task_id": task_id},
                    n_results=1
                )
                
                if task_results["documents"]:
                    for i, doc in enumerate(task_results["documents"][0]):
                        metadata = task_results["metadatas"][0][i]
                        context.append({
                            "type": "task",
                            "content": doc,
                            "task_id": metadata["task_id"],
                            "title": metadata["title"],
                            "status": metadata["status"],
                            "distance": task_results["distances"][0][i]
                        })
            
            # Search interactions
            interaction_results = self.interaction_collection.query(
                query_texts=[query],
                where={"session_id": session_id},
                n_results=min(limit, 2)
            )
            
            for i, doc in enumerate(interaction_results["documents"][0]):
                metadata = interaction_results["metadatas"][0][i]
                context.append({
                    "type": "interaction",
                    "content": doc,
                    "timestamp": metadata["timestamp"],
                    "tools_executed": json.loads(metadata["tools_executed"]),
                    "distance": interaction_results["distances"][0][i]
                })
            
            # Sort by relevance (distance)
            context.sort(key=lambda x: x["distance"])
            return context[:limit]
            
        except Exception as e:
            print(f"Error retrieving relevant context: {e}")
            return []
    
    async def get_task_history(
        self,
        session_id: str,
        status_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[TaskMemory]:
        """Get task history for a session."""
        if not self.task_collection:
            await self.initialize()
        
        try:
            where_clause = {"session_id": session_id}
            if status_filter:
                where_clause["status"] = status_filter
            
            results = self.task_collection.get(
                where=where_clause,
                limit=limit
            )
            
            tasks = []
            for i, metadata in enumerate(results["metadatas"]):
                tasks.append(TaskMemory(
                    task_id=metadata["task_id"],
                    session_id=metadata["session_id"],
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    title=metadata["title"],
                    description=metadata["description"],
                    plan=metadata.get("plan"),
                    tools_used=json.loads(metadata.get("tools_used", "[]")),
                    results=json.loads(metadata.get("results", "{}")),
                    status=metadata["status"],
                    metadata=metadata
                ))
            
            # Sort by timestamp (newest first)
            tasks.sort(key=lambda x: x.timestamp, reverse=True)
            return tasks
            
        except Exception as e:
            print(f"Error retrieving task history: {e}")
            return []
    
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        results: Dict[str, Any] = None
    ) -> bool:
        """Update task status and results."""
        if not self.task_collection:
            await self.initialize()
        
        try:
            # Get existing task
            existing = self.task_collection.get(ids=[task_id])
            if not existing["metadatas"]:
                return False
            
            metadata = existing["metadatas"][0].copy()
            metadata["status"] = status
            if results:
                metadata["results"] = json.dumps(results)
            metadata["updated_at"] = datetime.now().isoformat()
            
            # Update in collection
            self.task_collection.update(
                ids=[task_id],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            print(f"Error updating task status: {e}")
            return False
    
    async def store_error(
        self,
        session_id: str,
        error_context: Dict[str, Any]
    ) -> None:
        """Store error information for learning."""
        if not self.interaction_collection:
            await self.initialize()
        
        error_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        error_data = {
            "type": "error",
            "session_id": session_id,
            "error_context": json.dumps(error_context),
            "timestamp": timestamp.isoformat()
        }
        
        document = f"Error: {error_context.get('error', 'Unknown error')}"
        
        self.interaction_collection.add(
            documents=[document],
            metadatas=[error_data],
            ids=[error_id]
        )
    
    async def cleanup_old_memories(
        self,
        days_old: int = 30,
        keep_tasks: bool = True
    ) -> Dict[str, int]:
        """Clean up old memories to manage storage."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_iso = cutoff_date.isoformat()
        
        cleaned = {"conversations": 0, "interactions": 0, "tasks": 0}
        
        try:
            # Clean conversations
            if self.conversation_collection:
                conv_results = self.conversation_collection.get(
                    where={"timestamp": {"$lt": cutoff_iso}}
                )
                if conv_results["ids"]:
                    self.conversation_collection.delete(ids=conv_results["ids"])
                    cleaned["conversations"] = len(conv_results["ids"])
            
            # Clean interactions
            if self.interaction_collection:
                int_results = self.interaction_collection.get(
                    where={"timestamp": {"$lt": cutoff_iso}}
                )
                if int_results["ids"]:
                    self.interaction_collection.delete(ids=int_results["ids"])
                    cleaned["interactions"] = len(int_results["ids"])
            
            # Optionally clean completed tasks
            if not keep_tasks and self.task_collection:
                task_results = self.task_collection.get(
                    where={
                        "timestamp": {"$lt": cutoff_iso},
                        "status": {"$in": ["completed", "failed"]}
                    }
                )
                if task_results["ids"]:
                    self.task_collection.delete(ids=task_results["ids"])
                    cleaned["tasks"] = len(task_results["ids"])
            
            return cleaned
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return cleaned
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not all([self.conversation_collection, self.task_collection, self.interaction_collection]):
            await self.initialize()
        
        try:
            stats = {
                "conversations": self.conversation_collection.count(),
                "tasks": self.task_collection.count(),
                "interactions": self.interaction_collection.count(),
                "total_collections": 3
            }
            return stats
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}
