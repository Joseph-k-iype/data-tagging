"""
Classification service for mapping business terms to Preferred Business Terms (PBT).
"""

import logging
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import empty_checkpoint

# App imports
from app.config.settings import get_settings
from app.config.environment import get_os_env
from app.core.services.embeddings import get_embedding_service
from app.core.services.pbt_manager import get_pbt_manager
from app.core.services.confidence import get_confidence_service
from app.core.auth.auth_helper import get_azure_token_cached
from app.core.models.pbt import (
    PBT, MatchedPBT, MatchType, ConfidenceScore, 
    PBTClassificationResponse
)

logger = logging.getLogger(__name__)

# State type for the agent
class AgentState(Dict[str, Any]):
    """State dictionary for the LangGraph agent."""
    pass


class ClassificationService:
    """Service for classifying business terms against Preferred Business Terms (PBT)."""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ClassificationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the classification service."""
        if self._initialized:
            return
            
        self._initialized = True
        self.settings = get_settings()
        self.env = get_os_env()
        self.pbt_manager = get_pbt_manager()
        self.embedding_service = get_embedding_service()
        self.confidence_service = get_confidence_service()
        
        # LLM for classification
        self.llm = self._init_llm()
        
        # Memory for caching classification results
        self.memory_saver = MemorySaver()
        
        # Initialize the React agent with tools
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        
        # Request cache for avoiding duplicate classifications
        self.request_cache = {}
        
        logger.info("Classification service initialized")
    
    def _init_llm(self) -> AzureChatOpenAI:
        """Initialize the LLM for classification."""
        try:
            # Get Azure token
            token = get_azure_token_cached(
                tenant_id=self.settings.azure.tenant_id,
                client_id=self.settings.azure.client_id,
                client_secret=self.settings.azure.client_secret,
                scope="https://cognitiveservices.azure.com/.default"
            )
            
            if not token:
                logger.error("Failed to get Azure token for classification service")
                raise ValueError("Failed to get Azure token")
            
            # Create token provider function
            token_provider = lambda: token
            
            # Initialize Azure OpenAI client for LLM
            llm = AzureChatOpenAI(
                model_name=self.settings.azure.model_name,
                temperature=0.3,  # Lower temperature for more consistent classifications
                api_version=self.settings.azure.api_version,
                azure_endpoint=self.settings.azure.azure_endpoint,
                azure_ad_token_provider=token_provider
            )
            
            return llm
        
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _setup_tools(self) -> List[BaseTool]:
        """Set up tools for the React agent."""
        return [PBTClassifierTool(pbt_manager=self.pbt_manager)]
    
    def _create_agent(self):
        """Create the React agent with the LLM and tools."""
        # Define custom system message
        system_message = """
        You are an expert business terminology standardization system. Your task is to map user-provided terms 
        and descriptions to the organization's Preferred Business Terms (PBT).

        When a user provides a name and description, use the pbt_classifier tool to find the most appropriate 
        standard business terms from the database. 
        
        The tool will return both specific matches and broader category matches. Consider both types when making 
        your recommendation. For example, "drawdown client account number" might match to both specific terms 
        like "Account Number" and broader categories like "Account Identifier" or "Customer Account".
        
        Explain why the matches are appropriate, focusing on conceptual alignment rather than just keyword matching.
        Always mention both specific and broader matches in your response when available.
        
        You will also see if matches are made through synonyms, which indicates that the term uses alternative
        terminology for the same concept.
        
        If a term has a CDM (Conceptual Data Model) category, consider this in your matching. This is a higher-level
        categorization of the business term.
        
        You also have access to memory of previous classifications. Reference past similar classifications if relevant.
        """
        
        # Create the React agent with memory
        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=system_message,
            checkpointer=self.memory_saver
        )
        
        return agent
    
    def _get_memory_key(self, name: str, description: str) -> str:
        """Generate a consistent memory key for a classification request."""
        name_desc = f"{name.lower().strip()}:{description.lower().strip()}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, name_desc))
    
    async def _save_to_memory(self, session_id: str, name: str, description: str, result: Dict[str, Any]):
        """Save classification result to memory."""
        try:
            # Create a simplified memory entry
            memory_entry = {
                "input": {
                    "name": name,
                    "description": description,
                },
                "output": {
                    "best_match": result.get("best_match", {}).get("name", ""),
                    "best_match_id": result.get("best_match", {}).get("id", ""),
                    "confidence": result.get("confidence", {}).get("score", 0)
                },
                "timestamp": str(uuid.uuid1())  # Use timestamp for ordering
            }
            
            # Save to memory store
            memory_key = self._get_memory_key(name, description)
            namespace = f"classification:{memory_key}"
            
            # Create a proper config object with thread_id and checkpoint_ns
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": ""
                }
            }
            
            # Create an empty checkpoint and add our data to it
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"][namespace] = memory_entry
            
            # Use the put method with the correct signature
            self.memory_saver.put(
                config=config,
                checkpoint=checkpoint,
                metadata={"source": "input", "step": -1},
                new_versions={}
            )
            
            logger.info(f"Saved classification result to memory with ID {session_id} in namespace {namespace}")
        except Exception as e:
            logger.error(f"Error saving to memory: {e}")
    
    async def _get_from_memory(self, session_id: str, name: str, description: str) -> Optional[Dict[str, Any]]:
        """Retrieve classification from memory if it exists."""
        try:
            memory_key = self._get_memory_key(name, description)
            namespace = f"classification:{memory_key}"
            
            # Create a proper config object with thread_id and checkpoint_ns
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": ""
                }
            }
            
            # Use the get method to retrieve the checkpoint
            checkpoint = self.memory_saver.get(config)
            
            if checkpoint and namespace in checkpoint.get("channel_values", {}):
                return checkpoint["channel_values"][namespace]
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving from memory: {e}")
            return None
    
    async def _process_agent_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process the result from the agent."""
        try:
            # Extract the final message content
            messages = result.get("messages", [])
            final_message = messages[-1] if messages else None
            
            final_content = ""
            if hasattr(final_message, 'content'):
                final_content = final_message.content
            elif isinstance(final_message, dict) and 'content' in final_message:
                final_content = final_message['content']
            elif final_message is not None:
                final_content = str(final_message)
            
            # Extract tool results
            best_match = None
            specific_matches = []
            broader_matches = []
            
            for message in messages:
                # Find the tool message with classification results
                if (hasattr(message, 'type') and message.type == "tool" and 
                    (hasattr(message, 'name') and message.name == "pbt_classifier" or
                     hasattr(message, 'additional_kwargs') and 
                     message.additional_kwargs.get('name') == "pbt_classifier")):
                    
                    # Extract tool content
                    content = message.content if hasattr(message, 'content') else None
                    if content and isinstance(content, str):
                        try:
                            tool_content = json.loads(content)
                            
                            if tool_content.get("status") == "success":
                                best_match = tool_content.get("best_match")
                                specific_matches = tool_content.get("specific_matches", [])
                                broader_matches = tool_content.get("broader_matches", [])
                                break
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse tool content JSON")
            
            # Return processed result
            return {
                "agent_response": final_content,
                "best_match": best_match,
                "specific_matches": specific_matches,
                "broader_matches": broader_matches
            }
            
        except Exception as e:
            logger.error(f"Error processing agent result: {e}")
            return {}
    
    async def classify_with_llm(self, name: str, description: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Classify using LLM to select the best match from embedding candidates.
        
        Args:
            name: Name of the business term
            description: Description of the business term
            top_n: Number of results to return
            
        Returns:
            Classification result
        """
        try:
            # First get similar items with embeddings
            combined_input = f"{name} - {description}"
            similar_items = await self.pbt_manager.find_similar_items(
                combined_input, 
                top_n=top_n
            )
            
            if not similar_items:
                logger.warning(f"No similar items found for '{name}'")
                return {
                    "status": "error", 
                    "message": "No similar items found"
                }
            
            # Create a prompt for the LLM
            prompt = f"""
            You are an expert in business terminology. You need to select the most appropriate 
            Preferred Business Term (PBT) for the following input:
            
            Input Name: {name}
            Input Description: {description}
            
            Here are the candidate terms:
            
            """
            
            for i, item in enumerate(similar_items):
                prompt += f"""
                Option {i+1}:
                - ID: {item.id}
                - Name: {item.name}
                - Definition: {item.definition}
                - CDM Category: {item.cdm or 'N/A'}
                - Match Type: {item.match_type}
                - Similarity Score: {item.similarity_score:.2f}
                """
                
                if item.synonym_match:
                    prompt += f"- Matched via synonym: {item.matched_synonym}\n"
            
            prompt += """
            Select the BEST match based on conceptual alignment, not just keyword similarity.
            Return ONLY the ID of the best match without any explanation.
            """
            
            # Get the LLM's selection
            response = self.llm.invoke(prompt)
            selection_id = response.content.strip()
            
            # Find the selected item
            selected_item = next((item for item in similar_items if item.id == selection_id), None)
            
            # If not found, take the first item (highest similarity)
            if not selected_item:
                logger.warning(f"LLM returned ID '{selection_id}' that wasn't in candidates, using first match")
                selected_item = similar_items[0]
            
            # Get confidence score for the selected item
            confidence = await self.confidence_service.evaluate_confidence(
                combined_input, 
                selected_item
            )
            
            # Return result
            return {
                "status": "success",
                "best_match": selected_item,
                "specific_matches": [item for item in similar_items if item.match_type == MatchType.SPECIFIC],
                "broader_matches": [item for item in similar_items if item.match_type == MatchType.BROADER],
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return {"status": "error", "message": str(e)}
    
    async def classify_with_embeddings(self, name: str, description: str, top_n: int = 5, 
                                      include_broader_terms: bool = True) -> Dict[str, Any]:
        """
        Classify using embedding similarity only.
        
        Args:
            name: Name of the business term
            description: Description of the business term
            top_n: Number of results to return
            include_broader_terms: Whether to include broader terms
            
        Returns:
            Classification result
        """
        try:
            # Get similar items
            combined_input = f"{name} - {description}"
            similar_items = await self.pbt_manager.find_similar_items(
                combined_input, 
                top_n=top_n,
                include_broader_terms=include_broader_terms
            )
            
            if not similar_items:
                logger.warning(f"No similar items found for '{name}'")
                return {
                    "status": "error", 
                    "message": "No similar items found"
                }
            
            # Best match is the first item (highest similarity)
            best_match = similar_items[0]
            
            # Get confidence score for the best match
            confidence = await self.confidence_service.evaluate_confidence(
                combined_input, 
                best_match
            )
            
            # Return result
            return {
                "status": "success",
                "best_match": best_match,
                "specific_matches": [item for item in similar_items if item.match_type == MatchType.SPECIFIC],
                "broader_matches": [item for item in similar_items if item.match_type == MatchType.BROADER],
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in embedding classification: {e}")
            return {"status": "error", "message": str(e)}
    
    async def classify_with_agent(self, name: str, description: str) -> Dict[str, Any]:
        """
        Classify using the React agent with tools.
        
        Args:
            name: Name of the business term
            description: Description of the business term
            
        Returns:
            Classification result
        """
        try:
            # Generate a unique session ID for this classification
            session_id = str(uuid.uuid4())
            
            # Check if we have a similar classification in memory
            memory_entry = await self._get_from_memory(session_id, name, description)
            
            # Prepare input for the agent
            input_message = f"""
            Please map the following to standard Preferred Business Terms, including both specific and broader category matches:
            
            Name: {name}
            Description: {description}
            
            Provide both specific matches and broader category matches when available.
            """
            
            # If we have a memory entry, include it in the message
            if memory_entry:
                input_message += f"""
                
                I've previously classified similar input with the following result:
                Previous Input: {memory_entry['input']['name']} - {memory_entry['input']['description']}
                Previous Match: {memory_entry['output']['best_match']} (ID: {memory_entry['output']['best_match_id']})
                Previous Confidence: {memory_entry['output']['confidence']}
                """
            
            # Create a config for the agent
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": ""
                }
            }
            
            # Invoke the agent
            result = await asyncio.to_thread(
                self.agent.invoke,
                {"messages": [HumanMessage(content=input_message)]},
                config
            )
            
            # Process the agent result
            processed_result = await self._process_agent_result(result)
            
            if not processed_result.get("best_match"):
                # If agent couldn't find a match, fall back to embedding similarity
                logger.warning("Agent couldn't find a match, falling back to embedding similarity")
                embedding_result = await self.classify_with_embeddings(name, description)
                
                if embedding_result.get("status") == "error":
                    return embedding_result
                
                processed_result["best_match"] = embedding_result.get("best_match")
                processed_result["specific_matches"] = embedding_result.get("specific_matches", [])
                processed_result["broader_matches"] = embedding_result.get("broader_matches", [])
            
            # Convert best_match to MatchedPBT if it's a dict
            best_match = processed_result.get("best_match")
            if best_match and isinstance(best_match, dict):
                best_match = MatchedPBT(
                    id=str(best_match.get("id")),
                    name=best_match.get("PBT_NAME"),
                    definition=best_match.get("PBT_DEFINITION"),
                    cdm=best_match.get("CDM"),
                    match_type=MatchType.SPECIFIC,
                    similarity_score=best_match.get("similarity_score", 0.0),
                    synonym_match=best_match.get("synonym_match", False),
                    matched_synonym=best_match.get("matched_synonym")
                )
                processed_result["best_match"] = best_match
            
            # Get confidence score for the best match
            combined_input = f"{name} - {description}"
            confidence = await self.confidence_service.evaluate_confidence(
                combined_input, 
                best_match
            )
            
            # Add confidence to the result
            processed_result["confidence"] = confidence
            
            # Save to memory for future use
            await self._save_to_memory(session_id, name, description, processed_result)
            
            # Return result with status
            return {
                "status": "success",
                **processed_result
            }
            
        except Exception as e:
            logger.error(f"Error in agent classification: {e}")
            return {"status": "error", "message": str(e)}
    
    async def classify(self, name: str, description: str, method: str = "agent", 
                       include_broader_terms: bool = True, top_n: int = 5) -> PBTClassificationResponse:
        """
        Classify a business term using the specified method.
        
        Args:
            name: Name of the business term
            description: Description of the business term
            method: Classification method (embeddings, llm, agent)
            include_broader_terms: Whether to include broader terms
            top_n: Number of results to return
            
        Returns:
            Classification response
        """
        try:
            # Generate a request ID
            request_id = str(uuid.uuid4())
            
            # Check cache if enabled
            if self.settings.classification.cache_enabled:
                cache_key = f"{name}:{description}:{method}"
                if cache_key in self.request_cache:
                    cached_result = self.request_cache[cache_key]
                    cached_result.request_id = request_id
                    logger.info(f"Cache hit for '{name}' (method={method})")
                    return cached_result
            
            # Select the classification method
            if method == "embeddings":
                result = await self.classify_with_embeddings(
                    name, description, top_n, include_broader_terms
                )
            elif method == "llm":
                result = await self.classify_with_llm(name, description, top_n)
            elif method == "agent":
                result = await self.classify_with_agent(name, description)
            else:
                logger.error(f"Invalid classification method: {method}")
                return PBTClassificationResponse(
                    status="error",
                    request_id=request_id,
                    agent_response=f"Invalid classification method: {method}"
                )
            
            # Create the response
            response = PBTClassificationResponse(
                status=result.get("status", "error"),
                best_match=result.get("best_match"),
                specific_matches=result.get("specific_matches", []),
                broader_matches=result.get("broader_matches", []),
                confidence=result.get("confidence"),
                agent_response=result.get("agent_response"),
                request_id=request_id
            )
            
            # Cache the result if enabled
            if self.settings.classification.cache_enabled:
                cache_key = f"{name}:{description}:{method}"
                self.request_cache[cache_key] = response
                
                # Prune cache if too large (simple LRU implementation)
                if len(self.request_cache) > 1000:
                    # Remove oldest 20% of entries
                    remove_count = int(len(self.request_cache) * 0.2)
                    for _ in range(remove_count):
                        self.request_cache.pop(next(iter(self.request_cache)), None)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return PBTClassificationResponse(
                status="error",
                request_id=str(uuid.uuid4()),
                agent_response=f"Error: {str(e)}"
            )
    
    async def batch_classify(self, items: List[Dict[str, Any]], method: str = "agent") -> List[PBTClassificationResponse]:
        """
        Classify multiple business terms in batch.
        
        Args:
            items: List of business terms (each with name and description)
            method: Classification method (embeddings, llm, agent)
            
        Returns:
            List of classification responses
        """
        results = []
        
        # Process items in parallel with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent classifications
        
        async def classify_with_semaphore(item):
            async with semaphore:
                return await self.classify(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    method=method,
                    include_broader_terms=item.get("include_broader_terms", True),
                    top_n=item.get("top_n", 5)
                )
        
        # Create tasks
        tasks = [classify_with_semaphore(item) for item in items]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        return results


class PBTClassifierTool(BaseTool):
    """Tool for classifying business terms against PBT database."""
    
    name: str = "pbt_classifier"
    description: str = "Classifies user input against PBT (Preferred Business Terms) database entries"
    pbt_manager: Any
    
    def __init__(self, pbt_manager):
        """Initialize the tool."""
        super().__init__(
            name="pbt_classifier", 
            description="Classifies user input against PBT (Preferred Business Terms) database entries. Include parameter 'return_broader_terms=True' to get both specific and broader business terms."
        )
        self.pbt_manager = pbt_manager
    
    async def _arun(self, name: str, description: str, return_broader_terms: bool = True) -> str:
        """Run the tool asynchronously."""
        combined_input = f"{name} - {description}"
        similar_items = await self.pbt_manager.find_similar_items(
            combined_input, 
            top_n=5, 
            include_broader_terms=return_broader_terms
        )
        
        if not similar_items:
            return json.dumps({
                "status": "error", 
                "message": "No similar items found"
            })
        
        # Group results by match type
        specific_matches = [item for item in similar_items if item.match_type == MatchType.SPECIFIC]
        broader_matches = [item for item in similar_items if item.match_type == MatchType.BROADER]
        
        # Identify items matched via synonyms
        synonym_matches = [item for item in similar_items if item.synonym_match]
        
        # If no specific grouping, treat all as specific
        if not specific_matches and not broader_matches:
            specific_matches = similar_items
        
        # Format the result as a JSON string
        result = {
            "status": "success",
            "best_match": similar_items[0].dict() if similar_items else None,
            "specific_matches": [item.dict() for item in specific_matches],
            "broader_matches": [item.dict() for item in broader_matches],
            "similar_items": [item.dict() for item in similar_items]
        }
        
        # Add synonym match information if available
        if synonym_matches:
            result["synonym_matches"] = [
                {
                    "id": item.id,
                    "name": item.name,
                    "score": item.similarity_score,
                    "matched_synonym": item.matched_synonym
                } for item in synonym_matches
            ]
        
        return json.dumps(result)
    
    def _run(self, name: str, description: str, return_broader_terms: bool = True) -> str:
        """Run the tool synchronously by wrapping the async method."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._arun(name, description, return_broader_terms))


# Get the classification service instance
def get_classification_service() -> ClassificationService:
    """
    Get the classification service instance.
    
    Returns:
        ClassificationService: Classification service instance
    """
    return ClassificationService()