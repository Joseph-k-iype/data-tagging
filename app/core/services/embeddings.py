"""
Embedding service for generating and managing embeddings for the AI Tagging Service.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from azure.identity import get_bearer_token_provider
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from app.config.settings import get_settings
from app.config.environment import get_os_env
from app.core.auth.auth_helper import get_azure_token_cached

logger = logging.getLogger(__name__)

class Document(BaseModel):
    """Document model for embedding generation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    embedding: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    synonyms: List[str] = Field(default_factory=list)


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI."""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding service."""
        if self._initialized:
            return
            
        self._initialized = True
        self.settings = get_settings()
        self.env = get_os_env()
        
        # Initialize Azure OpenAI client for embeddings
        self.client = self._init_client()
        
        # Initialize the LLM for generating synonyms
        self.llm = self._init_llm()
        
        logger.info("Embedding service initialized")
    
    def _init_client(self) -> AzureOpenAI:
        """Initialize the Azure OpenAI client."""
        try:
            # Get Azure token
            token = get_azure_token_cached(
                tenant_id=self.settings.azure.tenant_id,
                client_id=self.settings.azure.client_id,
                client_secret=self.settings.azure.client_secret,
                scope="https://cognitiveservices.azure.com/.default"
            )
            
            if not token:
                logger.error("Failed to get Azure token for embedding service")
                raise ValueError("Failed to get Azure token")
            
            # Create token provider function
            token_provider = lambda: token
            
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                azure_endpoint=self.settings.azure.azure_endpoint,
                api_version=self.settings.azure.api_version,
                azure_ad_token_provider=token_provider
            )
            
            return client
        
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            raise
    
    def _init_llm(self) -> AzureChatOpenAI:
        """Initialize the LLM for generating synonyms."""
        try:
            # Get Azure token
            token = get_azure_token_cached(
                tenant_id=self.settings.azure.tenant_id,
                client_id=self.settings.azure.client_id,
                client_secret=self.settings.azure.client_secret,
                scope="https://cognitiveservices.azure.com/.default"
            )
            
            if not token:
                logger.error("Failed to get Azure token for LLM service")
                raise ValueError("Failed to get Azure token")
            
            # Create token provider function
            token_provider = lambda: token
            
            # Initialize Azure OpenAI client for LLM
            llm = AzureChatOpenAI(
                model_name=self.settings.azure.model_name,
                temperature=0.5,  # Good for creativity in synonyms
                max_tokens=300,
                api_version=self.settings.azure.api_version,
                azure_endpoint=self.settings.azure.azure_endpoint,
                azure_ad_token_provider=token_provider
            )
            
            return llm
        
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return None
    
    def generate_embedding(self, document: Document) -> Document:
        """
        Generate an embedding for a document.
        
        Args:
            document: Document to generate embedding for
            
        Returns:
            Document with embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.settings.azure.embedding_deployment_name,
                input=document.text,
                encoding_format="float"
            ).data[0].embedding
            
            document.embedding = response
            logger.debug(f"Generated embedding for document {document.id}")
            return document
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def batch_generate_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for a batch of documents.
        
        Args:
            documents: Documents to generate embeddings for
            
        Returns:
            Documents with embeddings
        """
        try:
            # Process in batches to avoid Azure API limits
            batch_size = 16
            processed_documents = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                texts = [doc.text for doc in batch]
                
                response = self.client.embeddings.create(
                    model=self.settings.azure.embedding_deployment_name,
                    input=texts,
                    encoding_format="float"
                )
                
                # Update documents with embeddings
                for j, embedding_data in enumerate(response.data):
                    batch[j].embedding = embedding_data.embedding
                
                processed_documents.extend(batch)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} of {(len(documents) - 1)//batch_size + 1}")
            
            return processed_documents
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def generate_synonyms(self, term_name: str, term_definition: str, max_synonyms: int = 10) -> List[str]:
        """
        Generate synonyms for a business term using the LLM.
        
        Args:
            term_name: Business term name
            term_definition: Business term definition
            max_synonyms: Maximum number of synonyms to generate
            
        Returns:
            List of synonyms
        """
        if not self.llm:
            logger.warning("LLM not initialized, cannot generate synonyms")
            return []
        
        try:
            prompt = f"""
            Generate {max_synonyms} alternative terms, phrases or synonyms that business users might use when referring to this business term:
            
            Term: {term_name}
            Definition: {term_definition}
            
            Provide ONLY a comma-separated list of alternative terms or phrases that a user might use when referring to this concept.
            These should be different ways to express the same concept, including industry jargon, abbreviations, and common variations.
            DO NOT provide explanations - ONLY the comma-separated list of terms.
            """
            
            response = self.llm.invoke(prompt)
            
            # Parse the response to extract the list of synonyms
            synonyms_text = response.content.strip()
            
            # Split by comma and clean up each synonym
            synonyms = [syn.strip() for syn in synonyms_text.split(',')]
            
            # Remove duplicates and empty strings
            synonyms = list(set([syn for syn in synonyms if syn]))
            
            logger.info(f"Generated {len(synonyms)} synonyms for '{term_name}'")
            return synonyms
            
        except Exception as e:
            logger.error(f"Error generating synonyms for '{term_name}': {e}")
            return []
    
    def get_langchain_compatible_embeddings(self):
        """
        Get a Langchain-compatible embeddings object for use with vector stores.
        
        Returns:
            A Langchain-compatible embeddings object
        """
        # Define a class that implements the Langchain embeddings interface
        class AzureEmbeddings:
            def __init__(self, service):
                self.service = service
            
            def embed_documents(self, texts):
                docs = [Document(id=str(i), text=text) for i, text in enumerate(texts)]
                embedded_docs = self.service.batch_generate_embeddings(docs)
                return [doc.embedding for doc in embedded_docs]
            
            def embed_query(self, text):
                doc = Document(id="query", text=text)
                embedded_doc = self.service.generate_embedding(doc)
                return embedded_doc.embedding
        
        return AzureEmbeddings(self)


# Get the embedding service instance
def get_embedding_service() -> EmbeddingService:
    """
    Get the embedding service instance.
    
    Returns:
        EmbeddingService: Embedding service instance
    """
    return EmbeddingService()