import json
import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Stage3SemanticRanker:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.company_embeddings = None
        self.company_profiles = None
        
    def rank_by_semantics(
        self,
        query: str,
        stage2_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        
        verified_companies = stage2_results.get("verified_companies", [])
        
        if not verified_companies:
            return {
                "semantic_scores": [],
                "ranked_companies": [],
                "query_embedding": None,
                "total_scored": 0,
            }
        
        query_embedding = self._encode_text(query)
        
        company_embeddings = []
        company_profiles = []
        
        for company in verified_companies:
            profile = self._build_company_profile(company)
            company_profiles.append(profile)
            embedding = self._encode_text(profile)
            company_embeddings.append(embedding)
        
        company_embeddings = np.array(company_embeddings)
        
        similarities = self._calculate_similarities(query_embedding, company_embeddings)
        
        semantic_scores = []
        for idx, (company, similarity) in enumerate(zip(verified_companies, similarities)):
            semantic_scores.append({
                "index": idx,
                "company": company.get("operational_name"),
                "semantic_score": float(similarity),
                "profile_preview": company_profiles[idx][:100] + "...",
            })
        
        semantic_scores.sort(key=lambda x: x["semantic_score"], reverse=True)
        
        ranked_companies = [verified_companies[score["index"]] for score in semantic_scores]
        
        return {
            "semantic_scores": semantic_scores,
            "ranked_companies": ranked_companies,
            "query_embedding": query_embedding,
            "total_scored": len(verified_companies),
            "company_profiles": {
                verified_companies[idx].get("operational_name"): company_profiles[idx]
                for idx in range(len(verified_companies))
            },
        }
    
    def _encode_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding
    
    def _build_company_profile(self, company: Dict[str, Any]) -> str:
        parts = []
        
        if company.get("operational_name"):
            parts.append(f"Company: {company['operational_name']}")
        
        if company.get("description"):
            parts.append(f"Description: {company['description']}")
        
        if company.get("core_offerings"):
            offerings = ", ".join(company["core_offerings"][:5])
            parts.append(f"Offerings: {offerings}")
        
        if company.get("target_markets"):
            markets = ", ".join(company["target_markets"][:5])
            parts.append(f"Markets: {markets}")
        
        if company.get("business_model"):
            parts.append(f"Model: {company['business_model']}")
        
        primary_naics = company.get("primary_naics")
        if primary_naics:
            if isinstance(primary_naics, str):
                try:
                    import ast
                    primary_naics = ast.literal_eval(primary_naics)
                except:
                    pass
            if isinstance(primary_naics, dict):
                label = primary_naics.get("label", "")
                if label:
                    parts.append(f"Industry: {label}")
        
        return " ".join(parts)
    
    def _calculate_similarities(
        self,
        query_embedding: np.ndarray,
        company_embeddings: np.ndarray,
    ) -> np.ndarray:
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, company_embeddings)[0]
        return similarities
    
    def explain_semantic_ranking(self, results: Dict[str, Any]) -> str:
        output = []
        output.append(f"Stage 3: Semantic Ranking")
        output.append(f"=========================")
        output.append(f"")
        output.append(f"Companies scored: {results.get('total_scored', 0)}")
        output.append(f"")
        
        semantic_scores = results.get("semantic_scores", [])
        if semantic_scores:
            output.append(f"Top Ranked (by semantic similarity):")
            for idx, score_info in enumerate(semantic_scores[:5], 1):
                output.append(f"  {idx}. {score_info['company']} (similarity: {score_info['semantic_score']:.4f})")
                output.append(f"     Profile: {score_info['profile_preview']}")
        
        return "\n".join(output)


def test_stage3():
    from query_parser import Stage1QueryParser
    from stage2_matcher import Stage2StructuredMatcher
    
    print("\n" + "="*80)
    print("STAGE 3: Semantic Ranker - Semantic Similarity Scoring")
    print("="*80)
    
    parser1 = Stage1QueryParser("companies (1).jsonl")
    matcher2 = Stage2StructuredMatcher()
    ranker3 = Stage3SemanticRanker()
    
    test_queries = [
        "Pharmaceutical companies in Switzerland",
        "Software companies in Germany",
        "Logistic companies in Romania",
    ]
    
    for query in test_queries:
        print(f"\n{'-'*80}")
        print(f"Query: {query}")
        
        stage1_results = parser1.parse_query(query)
        print(f"Stage 1: {stage1_results['match_count']} matches")
        
        stage2_results = matcher2.match_constraints(stage1_results['tokens'], stage1_results)
        print(f"Stage 2: {stage2_results['total_verified']} verified")
        
        stage3_results = ranker3.rank_by_semantics(query, stage2_results)
        print(f"Stage 3: {stage3_results['total_scored']} scored")
        
        print(ranker3.explain_semantic_ranking(stage3_results))
    
    print("\n" + "="*80)
    print("STAGE 3 - TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_stage3()
