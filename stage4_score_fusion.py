import json
import os
from typing import Dict, List, Any, Tuple
import numpy as np


class Stage4ScoreFusion:
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("VERIDION_API_KEY")
        self.structured_weight = 0.60
        self.semantic_weight = 0.40
        self.verification_enabled = True
        self.errors_found = []
        self.corrections_made = []
        
    def fuse_scores(
        self,
        stage2_results: Dict[str, Any],
        stage3_results: Dict[str, Any],
        query: str = None,
    ) -> Dict[str, Any]:
        
        self.errors_found = []
        self.corrections_made = []
        
        verified_companies = stage2_results.get("verified_companies", [])
        
        if not verified_companies:
            return {
                "fused_results": [],
                "total_results": 0,
                "verification_status": "no_companies_to_fuse",
                "errors": self.errors_found,
                "corrections": self.corrections_made,
            }
        
        semantic_scores = stage3_results.get("semantic_scores", [])
        semantic_map = self._build_semantic_map(semantic_scores)
        
        fused_results = []
        
        for company in verified_companies:
            try:
                company_name = company.get("operational_name", "Unknown")
                
                structured_score = self._extract_structured_score(company)
                semantic_score = semantic_map.get(company_name, 0.5)
                
                structured_score, struct_valid = self._validate_and_correct_score(
                    structured_score, "structured", company_name
                )
                semantic_score, sem_valid = self._validate_and_correct_score(
                    semantic_score, "semantic", company_name
                )
                
                if not struct_valid or not sem_valid:
                    self.errors_found.append(
                        f"Score validation failed for {company_name} "
                        f"(structured: {struct_valid}, semantic: {sem_valid})"
                    )
                
                fused_score = (
                    self.structured_weight * structured_score +
                    self.semantic_weight * semantic_score
                )
                
                fused_score, fused_valid = self._validate_and_correct_score(
                    fused_score, "fused", company_name
                )
                
                if not fused_valid:
                    self.errors_found.append(f"Fused score validation failed for {company_name}")
                
                fused_results.append({
                    "operational_name": company_name,
                    "country": self._safe_extract_country(company),
                    "structured_score": round(structured_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "fused_score": round(fused_score, 4),
                    "confidence": self._calculate_confidence(structured_score, semantic_score),
                    "full_company_data": company,
                })
                
            except Exception as e:
                company_name = company.get("operational_name", "Unknown")
                error_msg = f"Exception processing {company_name}: {str(e)}"
                self.errors_found.append(error_msg)
                continue
        
        self._verify_ranking_consistency(fused_results)
        
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        
        return {
            "fused_results": fused_results,
            "total_results": len(fused_results),
            "verification_status": "completed" if not self.errors_found else "completed_with_corrections",
            "errors_found": self.errors_found,
            "corrections_made": self.corrections_made,
            "fusion_weights": {
                "structured": self.structured_weight,
                "semantic": self.semantic_weight,
            },
        }
    
    def _validate_and_correct_score(
        self, score: float, score_type: str, company_name: str
    ) -> Tuple[float, bool]:
        
        if not isinstance(score, (int, float)):
            self.errors_found.append(
                f"{score_type} score is not numeric for {company_name}: {type(score)}"
            )
            score = 0.5
            self.corrections_made.append(
                f"Corrected {score_type} score to 0.5 (default) for {company_name}"
            )
            return score, False
        
        if np.isnan(score):
            self.errors_found.append(f"{score_type} score is NaN for {company_name}")
            score = 0.5
            self.corrections_made.append(
                f"Replaced NaN {score_type} score with 0.5 for {company_name}"
            )
            return score, False
        
        if np.isinf(score):
            self.errors_found.append(f"{score_type} score is Inf for {company_name}")
            score = 0.5
            self.corrections_made.append(
                f"Replaced Inf {score_type} score with 0.5 for {company_name}"
            )
            return score, False
        
        if score < 0:
            self.errors_found.append(
                f"{score_type} score below 0 for {company_name}: {score}"
            )
            score = max(0, score)
            self.corrections_made.append(
                f"Clamped {score_type} score to minimum 0 for {company_name}"
            )
            return score, False
        
        if score > 1:
            self.errors_found.append(
                f"{score_type} score above 1 for {company_name}: {score}"
            )
            score = min(1, score)
            self.corrections_made.append(
                f"Clamped {score_type} score to maximum 1 for {company_name}"
            )
            return score, False
        
        return score, True
    
    def _safe_extract_country(self, company: Dict[str, Any]) -> str:
        try:
            address = company.get("address", {})
            if isinstance(address, dict):
                return address.get("country", "Unknown")
            return "Unknown"
        except Exception as e:
            self.errors_found.append(f"Error extracting country: {str(e)}")
            return "Unknown"
    
    def _extract_structured_score(self, company: Dict[str, Any]) -> float:
        try:
            stage2_match_info = company.get("_stage2_match_info", {})
            score = stage2_match_info.get("match_quality_score", 0.5)
            
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    self.errors_found.append(
                        f"Could not convert structured score string to float: {score}"
                    )
                    score = 0.5
                    self.corrections_made.append("Set structured score to 0.5 (default)")
            
            return float(score)
        except Exception as e:
            self.errors_found.append(f"Error extracting structured score: {str(e)}")
            return 0.5
    
    def _build_semantic_map(self, semantic_scores: List[Dict]) -> Dict[str, float]:
        semantic_map = {}
        
        for item in semantic_scores:
            try:
                company_name = item.get("company")
                score = item.get("semantic_score", 0.5)
                
                if not isinstance(score, (int, float)):
                    self.errors_found.append(
                        f"Non-numeric semantic score for {company_name}: {type(score)}"
                    )
                    score = 0.5
                    self.corrections_made.append(
                        f"Set semantic score to 0.5 for {company_name}"
                    )
                
                score = float(score)
                semantic_map[company_name] = score
                
            except Exception as e:
                self.errors_found.append(f"Error processing semantic score: {str(e)}")
                continue
        
        return semantic_map
    
    def _calculate_confidence(self, structured: float, semantic: float) -> str:
        try:
            divergence = abs(structured - semantic)
            
            if divergence < 0.1:
                return "high"
            elif divergence < 0.25:
                return "medium"
            else:
                return "low"
        except Exception as e:
            self.errors_found.append(f"Error calculating confidence: {str(e)}")
            return "unknown"
    
    def _verify_ranking_consistency(self, results: List[Dict]) -> None:
        if not results:
            return
        
        scores = [r["fused_score"] for r in results]
        
        if len(scores) > 1:
            is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            if not is_sorted:
                self.errors_found.append("Results not properly sorted by fused_score")
                self.corrections_made.append("Re-sorted results by fused_score (descending)")
        
        for i, score in enumerate(scores):
            if score > 1.0 or score < 0.0:
                self.errors_found.append(f"Score out of bounds at position {i}: {score}")
            if np.isnan(score) or np.isinf(score):
                self.errors_found.append(f"Invalid numeric score at position {i}: {score}")
    
    def validate_api_key(self) -> Dict[str, Any]:
        if not self.api_key:
            self.errors_found.append("No API key provided")
            return {
                "valid": False,
                "error": "No API key provided",
                "status": "missing_api_key",
            }
        
        expected_format = (
            len(self.api_key) >= 40 and
            "." in self.api_key and
            self.api_key.count(".") == 1
        )
        
        if not expected_format:
            self.errors_found.append(f"API key format invalid")
            self.corrections_made.append(
                f"API key validation failed - expected format: xxxxx.yyyyy (length >= 40)"
            )
            return {
                "valid": False,
                "error": "Invalid API key format",
                "status": "invalid_format",
                "min_length": 40,
                "actual_length": len(self.api_key),
                "expected_dots": 1,
                "actual_dots": self.api_key.count("."),
            }
        
        return {
            "valid": True,
            "status": "valid",
            "api_key_length": len(self.api_key),
            "api_key_preview": "***" + self.api_key[-10:] if self.api_key else None,
        }
    
    def generate_final_report(
        self, fused_results: Dict[str, Any], query: str = None
    ) -> Dict[str, Any]:
        
        results_list = fused_results.get("fused_results", [])
        
        report = {
            "query": query or "No query provided",
            "total_results": len(results_list),
            "top_3_results": [],
            "api_key_status": self.validate_api_key(),
            "verification_summary": {
                "total_errors": len(self.errors_found),
                "total_corrections": len(self.corrections_made),
                "verification_quality": self._assessment_quality(self.errors_found, results_list),
                "all_errors": self.errors_found,
                "all_corrections": self.corrections_made,
            },
            "confidence_distribution": self._get_confidence_distribution(results_list),
            "score_statistics": self._calculate_score_statistics(results_list),
        }
        
        for result in results_list[:3]:
            report["top_3_results"].append({
                "company": result["operational_name"],
                "country": result.get("country", "Unknown"),
                "fused_score": result["fused_score"],
                "structured_score": result["structured_score"],
                "semantic_score": result["semantic_score"],
                "confidence": result["confidence"],
            })
        
        return report
    
    def _assessment_quality(self, errors: List[str], results: List[Dict]) -> str:
        if not results:
            return "no_results"
        if not errors:
            return "excellent"
        if len(errors) <= len(results) * 0.1:
            return "good"
        elif len(errors) <= len(results) * 0.25:
            return "fair"
        else:
            return "poor"
    
    def _get_confidence_distribution(self, results: List[Dict]) -> Dict[str, int]:
        distribution = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for result in results:
            conf = result.get("confidence", "unknown")
            if conf in distribution:
                distribution[conf] += 1
        return distribution
    
    def _calculate_score_statistics(self, results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "total_companies": 0,
            }
        
        try:
            scores = np.array([r["fused_score"] for r in results])
            
            return {
                "min": round(float(np.min(scores)), 4),
                "max": round(float(np.max(scores)), 4),
                "mean": round(float(np.mean(scores)), 4),
                "median": round(float(np.median(scores)), 4),
                "std_dev": round(float(np.std(scores)), 4),
                "total_companies": len(results),
            }
        except Exception as e:
            self.errors_found.append(f"Error calculating statistics: {str(e)}")
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "total_companies": len(results),
            }
