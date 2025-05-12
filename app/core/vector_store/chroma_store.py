"""
ChromaDB vector store implementation for the AI Tagging Service.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel
from app.config.settings import get_settings
from app.core.services.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """ChromaDB vector store for storing and retrieving embeddings."""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ChromaVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the ChromaDB vector store."""
        if self._initialized:
            return
            
        self._initialized = True
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        
        # Get embedding function for ChromaDB
        self.embedding_function = self.embedding_service.get_langchain_compatible_embeddings()
        
        # Create directory for persistence if it doesn't exist
        os.makedirs(self.settings.vector_db.persist_dir, exist_ok=True)
        
        # Create ChromaDB settings with telemetry disabled
        self.chroma_settings = ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
        
        # Initialize the vector store
        self.initialize()
        
        logger.info(f"ChromaDB vector store initialized with collection '{self.settings.vector_db.collection_name}' "
                    f"in '{self.settings.vector_db.persist_dir}'")
    
    def initialize(self):
        """Initialize the vector store."""
        try:
            self.vectorstore = Chroma(
                collection_name=self.settings.vector_db.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.settings.vector_db.persist_dir,
                client_settings=self.chroma_settings
            )
        except Exception as e:
            logger.error(f"Error initializing ChromaDB vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        try:
            if not ids:
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            
            # Process in batches to avoid overwhelming the database
            batch_size = 100
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                logger.info(f"Added batch {i//batch_size + 1} of {(len(documents) - 1)//batch_size + 1} to ChromaDB vector store")
            
            logger.info(f"Added total of {len(documents)} documents to ChromaDB vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        try:
            if not ids:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Process in batches
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                # Handle metadata batching
                batch_metadatas = None
                if metadatas:
                    batch_metadatas = metadatas[i:i + batch_size]
                
                self.vectorstore.add_texts(
                    texts=batch_texts, 
                    metadatas=batch_metadatas, 
                    ids=batch_ids
                )
                logger.info(f"Added text batch {i//batch_size + 1} of {(len(texts) - 1)//batch_size + 1} to ChromaDB vector store")
            
            logger.info(f"Added total of {len(texts)} texts to ChromaDB vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter to apply to the search
            
        Returns:
            List of similar documents
        """
        try:
            return self.vectorstore.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query with scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter to apply to the search
            
        Returns:
            List of (document, score) tuples
        """
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"Error searching vector store with scores: {e}")
            return []
    
    def max_marginal_relevance_search(self, query: str, k: int = 5, fetch_k: int = 20, 
                                     lambda_mult: float = 0.5, filter: Optional[Dict] = None) -> List[Document]:
        """
        Search for documents with maximal marginal relevance.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of documents to consider
            lambda_mult: Diversity vs. relevance tradeoff
            filter: Optional filter to apply to the search
            
        Returns:
            List of documents
        """
        try:
            return self.vectorstore.max_marginal_relevance_search(
                query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
            )
        except Exception as e:
            logger.error(f"Error with MMR search: {e}")
            return []
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """
        Get the vector store as a retriever.
        
        Args:
            search_type: Type of search to perform
            search_kwargs: Optional search kwargs
            
        Returns:
            Retriever
        """
        return self.vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if ids:
                self.vectorstore.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents from vector store")
                return True
            else:
                # Delete entire collection
                if hasattr(self.vectorstore, 'delete_collection'):
                    self.vectorstore.delete_collection()
                    # Reinitialize after deletion
                    self.initialize()
                else:
                    # Attempt to use client interface if available
                    try:
                        if hasattr(self.vectorstore, '_client'):
                            self.vectorstore._client.delete_collection(self.settings.vector_db.collection_name)
                            # Reinitialize after deletion
                            self.initialize()
                    except Exception as e:
                        logger.error(f"Error deleting collection via client: {e}")
                        return False
                
                logger.info(f"Deleted entire collection from vector store")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting from vector store: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if hasattr(self.vectorstore, '_client'):
                collection = self.vectorstore._client.get_collection(self.settings.vector_db.collection_name)
                count = collection.count()
                return {
                    "collection_name": self.settings.vector_db.collection_name,
                    "document_count": count,
                    "persist_directory": self.settings.vector_db.persist_dir
                }
            else:
                return {
                    "collection_name": self.settings.vector_db.collection_name,
                    "document_count": "unknown",
                    "persist_directory": self.settings.vector_db.persist_dir
                }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.settings.vector_db.collection_name,
                "error": str(e),
                "persist_directory": self.settings.vector_db.persist_dir
            }


# Get the ChromaDB vector store instance
def get_chroma_store() -> ChromaVectorStore:
    """
    Get the ChromaDB vector store instance.
    
    Returns:
        ChromaVectorStore: Vector store instance
    """
    return ChromaVectorStore()