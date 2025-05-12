"""
PBT Data Manager for loading, managing, and searching PBT data.
"""

import os
import logging
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from app.config.settings import get_settings
from app.core.services.embeddings import get_embedding_service
from app.core.vector_store.chroma_store import get_chroma_store
from app.core.models.pbt import PBT, MatchedPBT, MatchType, PBTStatistics

logger = logging.getLogger(__name__)

class PBTManager:
    """Manager for Preferred Business Terms (PBT) data."""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(PBTManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the PBT manager."""
        if self._initialized:
            return
            
        self._initialized = True
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_chroma_store()
        
        # PBT data
        self.pbt_data = []
        self.concept_hierarchy = {}
        
        logger.info("PBT Manager initialized")
    
    async def load_csv(self, csv_path: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
        """
        Load PBT data from CSV file and store in vector store with synonyms.
        
        Args:
            csv_path: Path to the CSV file
            reload: Whether to reload the data if already loaded
            
        Returns:
            Dict with load status and count
        """
        try:
            # Use provided path or default from settings
            file_path = csv_path or self.settings.pbt_csv_path
            
            # Check if data is already loaded and reload flag is not set
            if not reload and len(self.pbt_data) > 0:
                logger.info("PBT data already loaded. Use reload=True to force reload.")
                return {
                    "status": "success", 
                    "message": "PBT data already loaded", 
                    "total_loaded": len(self.pbt_data)
                }
            
            # Ensure the file exists
            if not os.path.exists(file_path):
                logger.error(f"CSV file not found: {file_path}")
                return {"status": "error", "message": f"CSV file not found: {file_path}", "total_loaded": 0}
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_columns = {'id', 'PBT_NAME', 'PBT_DEFINITION'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                logger.error(f"CSV file missing required columns: {missing}")
                return {
                    "status": "error", 
                    "message": f"CSV file missing required columns: {missing}", 
                    "total_loaded": 0
                }
            
            # Add CDM column if not already present
            if 'CDM' not in df.columns:
                logger.warning("CSV file does not contain CDM column. Adding empty CDM values.")
                df['CDM'] = None
            
            # Convert DataFrame to list of dictionaries
            self.pbt_data = df.to_dict('records')
            
            # Delete all existing data in the vector store
            self.vector_store.delete()
            
            # Create documents for embedding with synonyms
            all_docs = []
            all_ids = []
            
            # Process in smaller batches for synonym generation
            synonym_batch_size = 10
            for i in range(0, len(self.pbt_data), synonym_batch_size):
                batch = self.pbt_data[i:i + synonym_batch_size]
                batch_docs = []
                batch_ids = []
                
                logger.info(f"Generating synonyms for batch {i//synonym_batch_size + 1} of {(len(self.pbt_data) - 1)//synonym_batch_size + 1}")
                
                for item in batch:
                    # Generate synonyms for this term
                    synonyms = self.embedding_service.generate_synonyms(
                        term_name=item['PBT_NAME'],
                        term_definition=item['PBT_DEFINITION']
                    )
                    
                    # Convert synonyms to a comma-separated string
                    synonyms_str = ", ".join(synonyms) if synonyms else ""
                    
                    # Combined text including the term, definition, and CDM (if available)
                    combined_text = f"{item['PBT_NAME']} - {item['PBT_DEFINITION']}"
                    if 'CDM' in item and item['CDM']:
                        combined_text += f" - {item['CDM']}"
                    
                    # Create metadata
                    metadata = {
                        'id': str(item['id']),
                        'name': item['PBT_NAME'],
                        'definition': item['PBT_DEFINITION'],
                        'cdm': item.get('CDM', ''),
                        'synonyms_str': synonyms_str,
                        'synonym_count': len(synonyms)
                    }
                    
                    # Create a document
                    doc = Document(
                        page_content=combined_text,
                        metadata=metadata
                    )
                    
                    batch_docs.append(doc)
                    batch_ids.append(str(item['id']))
                
                all_docs.extend(batch_docs)
                all_ids.extend(batch_ids)
            
            # Add documents to vector store
            self.vector_store.add_documents(all_docs, ids=all_ids)
            
            # Build concept hierarchy
            await self._build_concept_hierarchy()
            
            logger.info(f"Loaded {len(self.pbt_data)} PBT records with synonyms from CSV into vector store")
            
            return {
                "status": "success", 
                "message": f"Loaded {len(self.pbt_data)} PBT records", 
                "total_loaded": len(self.pbt_data)
            }
        
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return {"status": "error", "message": f"Error loading CSV: {str(e)}", "total_loaded": 0}
    
    async def _build_concept_hierarchy(self):
        """
        Build a concept hierarchy based on the similarity between terms.
        This identifies broader and more specific terms.
        """
        try:
            # Get all documents with embeddings
            all_docs = self.vector_store.similarity_search("", k=len(self.pbt_data) + 1)
            
            # Prepare for similarity calculation
            embeddings = []
            doc_ids = []
            doc_texts = []
            
            for doc in all_docs:
                doc_id = doc.metadata.get('id')
                if doc_id:
                    doc_ids.append(doc_id)
                    doc_texts.append(doc.page_content)
                    
                    # Create a dummy document to get the embedding
                    query_text = doc.page_content
                    doc_embedding = self.embedding_service.generate_embedding(
                        self.embedding_service.Document(id=doc_id, text=query_text)
                    ).embedding
                    
                    embeddings.append(doc_embedding)
            
            # Calculate similarity matrix
            similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
            
            for i in range(len(embeddings)):
                for j in range(len(embeddings)):
                    if i != j:  # Skip self-comparison
                        similarity_matrix[i, j] = cosine_similarity(
                            [embeddings[i]], 
                            [embeddings[j]]
                        )[0][0]
            
            # Calculate average similarities (higher means more general)
            avg_similarities = np.mean(similarity_matrix, axis=1)
            
            # Calculate term specificity factors
            term_generality = {}
            term_lengths = []
            
            for i, doc_id in enumerate(doc_ids):
                # Find the corresponding PBT data item
                item = next((item for item in self.pbt_data if str(item['id']) == doc_id), None)
                if item:
                    # Calculate term specificity based on term length and average similarity
                    name_length = len(item['PBT_NAME'].split())
                    term_lengths.append(name_length)
                    
                    term_generality[doc_id] = {
                        'avg_similarity': float(avg_similarities[i]),
                        'name_length': name_length,
                        'id': item['id'],
                        'name': item['PBT_NAME'],
                        'cdm': item.get('CDM', '')
                    }
            
            # Normalize term lengths
            max_length = max(term_lengths) if term_lengths else 1
            for term_id, data in term_generality.items():
                # Higher score means more general
                data['generality_score'] = data['avg_similarity'] * (1 - (data['name_length'] / max_length))
            
            # Sort terms by generality score
            sorted_terms = sorted(
                term_generality.items(), 
                key=lambda x: x[1]['generality_score'], 
                reverse=True
            )
            
            # Take the top 20% as broader terms
            broader_terms_count = max(1, int(len(sorted_terms) * 0.2))
            broader_terms = [term[0] for term in sorted_terms[:broader_terms_count]]
            
            # Store in concept hierarchy
            self.concept_hierarchy = {
                'broader_terms': [item for item in self.pbt_data 
                                if str(item['id']) in broader_terms],
                'term_generality': term_generality
            }
            
            logger.info(f"Built concept hierarchy with {len(broader_terms)} broader terms")
        
        except Exception as e:
            logger.error(f"Error building concept hierarchy: {e}")
            self.concept_hierarchy = {'broader_terms': [], 'term_generality': {}}
    
    async def find_similar_items(self, query_text: str, top_n: int = 5, include_broader_terms: bool = True) -> List[MatchedPBT]:
        """
        Find the most similar PBT items to the query text.
        
        Args:
            query_text: Text to search for
            top_n: Number of results to return
            include_broader_terms: Whether to include broader terms
            
        Returns:
            List of matched PBT items
        """
        try:
            # Get similar documents with scores
            doc_score_pairs = self.vector_store.similarity_search_with_score(
                query_text, 
                k=top_n * 2  # Get more results initially
            )
            
            # Process results to check for synonym matches
            exact_matches = []
            
            for doc, score in doc_score_pairs:
                doc_id = doc.metadata.get('id')
                synonyms_str = doc.metadata.get('synonyms_str', "")
                
                # Split synonyms string into a list
                synonyms = [s.strip() for s in synonyms_str.split(',')] if synonyms_str else []
                
                # Check for synonym matches
                query_terms = set(query_text.lower().split())
                
                synonym_match = False
                matched_synonym = None
                
                for synonym in synonyms:
                    if not synonym:
                        continue
                    synonym_terms = set(synonym.lower().split())
                    if query_terms.intersection(synonym_terms):
                        synonym_match = True
                        matched_synonym = synonym
                        break
                
                # Find the corresponding PBT data item
                item = next((dict(item) for item in self.pbt_data if str(item['id']) == doc_id), None)
                
                if item:
                    # Create MatchedPBT object
                    matched_pbt = MatchedPBT(
                        id=str(item['id']),
                        name=item['PBT_NAME'],
                        definition=item['PBT_DEFINITION'],
                        cdm=item.get('CDM'),
                        match_type=MatchType.SPECIFIC,
                        similarity_score=float(score),
                        synonym_match=synonym_match,
                        matched_synonym=matched_synonym if synonym_match else None
                    )
                    
                    # Boost score for synonym matches
                    if synonym_match:
                        matched_pbt.similarity_score *= 1.2  # 20% boost
                    
                    exact_matches.append(matched_pbt)
            
            # Sort by score and take top_n
            exact_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            specific_results = exact_matches[:top_n]
            
            final_results = specific_results
            
            # Include broader terms if requested
            if include_broader_terms and self.concept_hierarchy.get('broader_terms'):
                broader_matches = []
                
                for broader_term in self.concept_hierarchy.get('broader_terms', []):
                    # Skip terms already in specific results
                    if any(r.id == str(broader_term['id']) for r in specific_results):
                        continue
                    
                    # Create MatchedPBT object
                    matched_pbt = MatchedPBT(
                        id=str(broader_term['id']),
                        name=broader_term['PBT_NAME'],
                        definition=broader_term['PBT_DEFINITION'],
                        cdm=broader_term.get('CDM'),
                        match_type=MatchType.BROADER,
                        similarity_score=0.5,  # Default score for broader terms
                        synonym_match=False
                    )
                    
                    broader_matches.append(matched_pbt)
                
                # Sort broader matches by name length (shorter names are usually more general)
                broader_matches.sort(key=lambda x: len(x.name.split()), reverse=False)
                
                # Take up to 3 broader matches
                broader_matches = broader_matches[:3]
                
                # Add broader matches to results
                final_results = specific_results + broader_matches
            
            return final_results
        
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []
    
    async def get_pbt_by_id(self, pbt_id: str) -> Optional[PBT]:
        """
        Get a PBT by its ID.
        
        Args:
            pbt_id: PBT ID
            
        Returns:
            PBT if found, None otherwise
        """
        try:
            item = next((item for item in self.pbt_data if str(item['id']) == pbt_id), None)
            if item:
                # Get synonyms from vector store
                filter_dict = {"id": pbt_id}
                docs = self.vector_store.similarity_search("", k=1, filter=filter_dict)
                
                synonyms = []
                if docs:
                    synonyms_str = docs[0].metadata.get('synonyms_str', "")
                    synonyms = [s.strip() for s in synonyms_str.split(',')] if synonyms_str else []
                
                return PBT(
                    id=str(item['id']),
                    PBT_NAME=item['PBT_NAME'],
                    PBT_DEFINITION=item['PBT_DEFINITION'],
                    CDM=item.get('CDM'),
                    synonyms=synonyms
                )
            return None
        
        except Exception as e:
            logger.error(f"Error getting PBT by ID: {e}")
            return None
    
    async def get_statistics(self) -> PBTStatistics:
        """
        Get statistics about the PBT data.
        
        Returns:
            Statistics about the PBT data
        """
        try:
            total_count = len(self.pbt_data)
            
            # Get vector store stats
            vector_store_stats = self.vector_store.get_collection_stats()
            indexed_count = vector_store_stats.get("document_count", 0)
            if isinstance(indexed_count, str):
                indexed_count = total_count  # Fallback if we couldn't get accurate count
            
            # Count PBTs by CDM category
            cdm_categories = {}
            for item in self.pbt_data:
                cdm = item.get('CDM', '')
                if not cdm:
                    cdm = "Uncategorized"
                if cdm in cdm_categories:
                    cdm_categories[cdm] += 1
                else:
                    cdm_categories[cdm] = 1
            
            # Count PBTs with synonyms
            has_synonyms_count = 0
            total_synonyms = 0
            
            # Query the vector store for synonym counts
            for item in self.pbt_data:
                filter_dict = {"id": str(item['id'])}
                docs = self.vector_store.similarity_search("", k=1, filter=filter_dict)
                
                if docs:
                    synonym_count = docs[0].metadata.get('synonym_count', 0)
                    if synonym_count > 0:
                        has_synonyms_count += 1
                        total_synonyms += synonym_count
            
            # Calculate average synonyms per PBT
            avg_synonyms = total_synonyms / max(1, total_count)
            
            # Get top CDM categories
            top_cdm = sorted(
                [{"category": k, "count": v} for k, v in cdm_categories.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:5]
            
            return PBTStatistics(
                total_pbt_count=total_count,
                indexed_count=indexed_count,
                cdm_categories=cdm_categories,
                has_synonyms_count=has_synonyms_count,
                average_synonyms_per_pbt=avg_synonyms,
                top_cdm_categories=top_cdm
            )
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return PBTStatistics(
                total_pbt_count=0,
                indexed_count=0,
                cdm_categories={},
                has_synonyms_count=0,
                average_synonyms_per_pbt=0,
                top_cdm_categories=[]
            )


# Get the PBT manager instance
def get_pbt_manager() -> PBTManager:
    """
    Get the PBT manager instance.
    
    Returns:
        PBTManager: PBT manager instance
    """
    return PBTManager()