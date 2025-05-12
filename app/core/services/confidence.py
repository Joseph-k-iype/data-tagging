"""
Confidence evaluation service for assessing match quality between business terms and PBTs.
"""

import logging
import json
import uuid
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import empty_checkpoint

from app.config.settings import get_settings
from app.config.environment import get_os_env
from app.core.auth.auth_helper import get_azure_token_cached
from app.core.models.pbt import ConfidenceScore, MatchedPBT

logger = logging.getLogger(__name__)

class ConfidenceService:
    """Service for evaluating confidence of matches between business terms and PBTs."""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfidenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the confidence service."""
        if self._initialized:
            return
            
        self._initialized = True
        self.settings = get_settings()
        self.env = get_os_env()
        
        # LLM for confidence evaluation
        self.llm = self._init_llm()
        
        # Chain for confidence evaluation
        self.chain = self._create_confidence_chain()
        
        # Memory for caching confidence evaluations
        self.memory_saver = MemorySaver()
        
        # In-memory cache for fast lookups
        self.cache = {}
        
        logger.info("Confidence service initialized")
    
    def _init_llm(self) -> AzureChatOpenAI:
        """Initialize the LLM for confidence evaluation."""
        try:
            # Get Azure token
            token = get_azure_token_cached(
                tenant_id=self.settings.azure.tenant_id,
                client_id=self.settings.azure.client_id,
                client_secret=self.settings.azure.client_secret,
                scope="https://cognitiveservices.azure.com/.default"
            )
            
            if not token:
                logger.error("Failed to get Azure token for confidence service")
                raise ValueError("Failed to get Azure token")
            
            # Create token provider function
            token_provider = lambda: token
            
            # Initialize Azure OpenAI client for LLM
            llm = AzureChatOpenAI(
                model_name=self.settings.azure.model_name,
                temperature=0.2,  # Lower temperature for more consistent confidence scores
                api_version=self.settings.azure.api_version,
                azure_endpoint=self.settings.azure.azure_endpoint,
                azure_ad_token_provider=token_provider
            )
            
            return llm
        
        except Exception as e:
            logger.error(f"Error initializing LLM for confidence evaluation: {e}")
            raise
    
    def _create_confidence_chain(self):
        """Create the confidence evaluation chain."""
        # Define prompt for confidence evaluation
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are an expert system that evaluates the confidence of matches between user-provided terms and standard
            Preferred Business Terms (PBT). Analyze the semantic similarity, contextual relevance, and overall 
            appropriateness of the match.
            
            Provide a confidence score between 0 and 100, where:
            - 0-20: Very low confidence. The match seems arbitrary or incorrect.
            - 21-40: Low confidence. There's a vague relationship but likely not the best match.
            - 41-60: Moderate confidence. There's a reasonable connection but potentially better alternatives.
            - 61-80: High confidence. The match is strong and likely appropriate.
            - 81-100: Very high confidence. The match is excellent and almost certainly correct.
            
            Return your evaluation as a JSON object with the following fields:
            - score: (number between 0-100)
            - explanation: (string explaining your reasoning)
            """),
            HumanMessage(content="""
            User Input: {user_input}
            
            Matched PBT: 
            ID: {pbt_id}
            Name: {pbt_name}
            Definition: {pbt_definition}
            CDM: {pbt_cdm}
            
            Additional Information:
            - Match Type: {match_type}
            - Was matched via synonym: {synonym_match}
            
            Evaluate the confidence of this match.
            """)
        ])
        
        # Create the chain
        return prompt | self.llm
    
    def _get_memory_key(self, user_input: str, pbt_id: str) -> str:
        """Generate a consistent memory key for retrieving past evaluations."""
        return f"confidence:{user_input.strip().lower()}:{pbt_id}"
    
    async def evaluate_confidence(self, user_input: str, 
                                 pbt_match: Union[Dict[str, Any], MatchedPBT]) -> ConfidenceScore:
        """
        Evaluate the confidence of a match between user input and a PBT.
        
        Args:
            user_input: User input (name and description)
            pbt_match: Matched PBT (either a dict or MatchedPBT object)
            
        Returns:
            ConfidenceScore with score and explanation
        """
        try:
            # Convert MatchedPBT to dict if needed
            if isinstance(pbt_match, MatchedPBT):
                pbt_dict = {
                    "id": pbt_match.id,
                    "name": pbt_match.name,
                    "definition": pbt_match.definition,
                    "cdm": pbt_match.cdm,
                    "match_type": pbt_match.match_type,
                    "synonym_match": pbt_match.synonym_match
                }
            else:
                # Normalize the dict keys
                pbt_dict = {
                    "id": pbt_match.get("id") or pbt_match.get("ID"),
                    "name": pbt_match.get("name") or pbt_match.get("PBT_NAME"),
                    "definition": pbt_match.get("definition") or pbt_match.get("PBT_DEFINITION"),
                    "cdm": pbt_match.get("cdm") or pbt_match.get("CDM"),
                    "match_type": pbt_match.get("match_type", "specific"),
                    "synonym_match": pbt_match.get("synonym_match", False)
                }
            
            # Generate a memory key for this evaluation
            memory_key = self._get_memory_key(user_input, str(pbt_dict["id"]))
            
            # Check in-memory cache first
            if memory_key in self.cache:
                logger.info(f"Found cached confidence evaluation in memory: {memory_key}")
                return self.cache[memory_key]
            
            # Create a session ID for memory storage
            session_id = "confidence_evaluations"
            
            # Create a proper config object with thread_id and checkpoint_ns
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": ""
                }
            }
            
            # Try to get from persistent memory
            try:
                checkpoint = self.memory_saver.get(config)
                
                if checkpoint and memory_key in checkpoint.get("channel_values", {}):
                    cached_result = checkpoint["channel_values"][memory_key]
                    logger.info(f"Found cached confidence evaluation in persistent memory: {memory_key}")
                    
                    # Parse the cached result
                    score = cached_result.get("score", 0)
                    explanation = cached_result.get("explanation", "")
                    
                    # Create ConfidenceScore and cache it
                    confidence_score = ConfidenceScore(score=score, explanation=explanation)
                    self.cache[memory_key] = confidence_score
                    
                    return confidence_score
            except Exception as mem_error:
                logger.warning(f"Error retrieving from memory: {mem_error}")
            
            # Call the LLM chain for confidence evaluation
            llm_response = await self.chain.ainvoke({
                "user_input": user_input,
                "pbt_id": pbt_dict["id"],
                "pbt_name": pbt_dict["name"],
                "pbt_definition": pbt_dict["definition"],
                "pbt_cdm": pbt_dict["cdm"] or "N/A",
                "match_type": pbt_dict["match_type"],
                "synonym_match": "Yes" if pbt_dict["synonym_match"] else "No"
            })
            
            # Extract JSON data from the response
            content = llm_response.content
            
            # Parse the JSON response
            try:
                # Try direct JSON parsing first
                result = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON using regex
                import re
                json_pattern = r'\{.*\}'
                match = re.search(json_pattern, content, re.DOTALL)
                
                if match:
                    try:
                        result = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        # If regex extraction fails, create a default result
                        result = {
                            "score": 50,
                            "explanation": "Could not parse confidence score from LLM output. Using default medium confidence."
                        }
                else:
                    # If no JSON-like structure found, create a default result
                    result = {
                        "score": 50,
                        "explanation": "Could not parse confidence score from LLM output. Using default medium confidence."
                    }
            
            # Ensure the result has the expected fields
            score = result.get("score", 50)
            if isinstance(score, str):
                try:
                    score = int(score)
                except ValueError:
                    score = 50
            
            explanation = result.get("explanation", "No explanation provided")
            
            # Create a ConfidenceScore object
            confidence_score = ConfidenceScore(score=score, explanation=explanation)
            
            # Save to memory
            try:
                # Create or update the checkpoint
                checkpoint = self.memory_saver.get(config) or empty_checkpoint()
                checkpoint["channel_values"][memory_key] = {
                    "score": score,
                    "explanation": explanation
                }
                
                # Use the put method with the correct signature
                self.memory_saver.put(
                    config=config,
                    checkpoint=checkpoint,
                    metadata={"source": "input", "step": -1},
                    new_versions={}
                )
                
                logger.info(f"Saved confidence evaluation to memory: {memory_key}")
            except Exception as mem_error:
                logger.warning(f"Error saving to memory: {mem_error}")
            
            # Cache the result in memory
            self.cache[memory_key] = confidence_score
            
            return confidence_score
            
        except Exception as e:
            logger.error(f"Error in confidence evaluation: {e}")
            return ConfidenceScore(
                score=50,
                explanation=f"Error evaluating confidence: {str(e)}"
            )


# Get the confidence service instance
def get_confidence_service() -> ConfidenceService:
    """
    Get the confidence service instance.
    
    Returns:
        ConfidenceService: Confidence service instance
    """
    return ConfidenceService()