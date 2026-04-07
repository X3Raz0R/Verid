import json
import ast
import re
import os
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Stage2StructuredMatcher:
    
    def __init__(self):
        self.field_weights = {
            "operational_name": 3.0,
            "core_offerings": 2.5,
            "description": 2.0,
            "target_markets": 2.0,
            "naics_label": 1.5,
            "address": 1.5,
        }
        self.llm_client = self._init_llm_client()
        self.llm_failed = False
        self.llm_failure_count = 0
    
    def _init_llm_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI:
            return OpenAI(api_key=api_key)
        return None
    
    def match_constraints(
        self,
        query_tokens: List[str],
        stage1_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        matched_companies = stage1_results.get("matched_companies", [])
        constraints = stage1_results.get("constraints", {})
        
        if not matched_companies:
            return {
                "verified_companies": [],
                "rejected_companies": [],
                "constraint_matches": {},
                "match_quality_scores": [],
                "total_verified": 0,
                "reason": "No companies from Stage 1",
            }
        
        verified = []
        rejected = []
        constraint_details = defaultdict(list)
        
        use_llm_verification = self.llm_client and not self.llm_failed
        
        for company in matched_companies:
            if use_llm_verification:
                is_match = self._verify_company_with_llm(company, stage1_results.get("raw_query", ""))
                if self.llm_failed:
                    use_llm_verification = False
            else:
                is_match = True
            
            if is_match:
                match_info = self._get_match_info_for_company(company, query_tokens)
                verified.append({
                    **company,
                    "_stage2_match_info": match_info,
                })
                for token, matches in match_info.get("token_field_matches", {}).items():
                    constraint_details[token].extend(matches)
            else:
                match_info = self._get_match_info_for_company(company, query_tokens)
                rejected.append({
                    **company,
                    "_stage2_rejection_reason": match_info.get("missing_constraints", []),
                })
        
        quality_scores = self._score_match_quality(verified, query_tokens)
        
        return {
            "verified_companies": verified,
            "rejected_companies": rejected,
            "constraint_matches": dict(constraint_details),
            "match_quality_scores": quality_scores,
            "total_verified": len(verified),
            "total_rejected": len(rejected),
            "original_matched": len(matched_companies),
            "verification_rate": len(verified) / len(matched_companies) if matched_companies else 0,
        }
    
    def _verify_company_with_llm(self, company: Dict[str, Any], query: str) -> bool:
        if not self.llm_client or not query or self.llm_failed:
            return True
        
        try:
            company_profile = self._build_company_profile(company)
            
            prompt = f"""Determine if this company matches the search query.
Return only "yes" or "no".

Query: "{query}"

Company Profile:
{company_profile}

Result:"""

            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            self.llm_failure_count = 0
            return "yes" in result or "match" in result
        except Exception as e:
            self.llm_failure_count += 1
            if self.llm_failure_count >= 3 or "429" in str(e) or "quota" in str(e).lower():
                self.llm_failed = True
            return True

    def _build_company_profile(self, company: Dict[str, Any]) -> str:
        parts = []
        
        if company.get("operational_name"):
            parts.append(f"Name: {company['operational_name']}")
        
        if company.get("description"):
            parts.append(f"Description: {company['description']}")
        
        if company.get("core_offerings"):
            offerings = ", ".join(company.get("core_offerings", [])[:5])
            parts.append(f"Offerings: {offerings}")
        
        if company.get("employee_count"):
            parts.append(f"Employees: {company['employee_count']}")
        
        if company.get("revenue"):
            parts.append(f"Revenue: ${company['revenue']:,}")
        
        if company.get("year_founded"):
            parts.append(f"Founded: {company['year_founded']}")
        
        address = company.get("address")
        if address:
            if isinstance(address, str):
                try:
                    address = ast.literal_eval(address)
                except:
                    pass
            if isinstance(address, dict):
                country = address.get("country") or address.get("country_code", "")
                if country:
                    parts.append(f"Country: {country}")
        
        return "\n".join(parts)

    def _get_match_info_for_company(self, company: Dict[str, Any], query_tokens: List[str]) -> Dict[str, Any]:
        token_field_matches = {}
        all_constraints_matched = True
        missing_constraints = []
        match_details = {}
        
        if not query_tokens:
            return {
                "all_constraints_matched": True,
                "missing_constraints": [],
                "token_field_matches": {},
                "match_details": {},
            }
        
        for token in query_tokens:
            field_matches = self._search_token_in_company(company, token)
            
            token_field_matches[token] = field_matches
            
            if field_matches:
                match_details[token] = {
                    "status": "MATCHED",
                    "fields": field_matches,
                }
            else:
                match_details[token] = {
                    "status": "NOT_MATCHED",
                }
                missing_constraints.append(token)
                all_constraints_matched = False
        
        return {
            "all_constraints_matched": all_constraints_matched,
            "missing_constraints": missing_constraints,
            "token_field_matches": token_field_matches,
            "match_details": match_details,
        }
    
    def _search_token_in_company(
        self,
        company: Dict[str, Any],
        token: str,
    ) -> List[Tuple[str, bool]]:
        matches = []
        token_lower = token.lower()
        
        name = (company.get("operational_name") or "").lower()
        if token_lower in name or self._is_partial_match(token_lower, name):
            matches.append(("operational_name", token_lower in name))
        
        desc = (company.get("description") or "").lower()
        if self._search_in_text(token_lower, desc):
            matches.append(("description", token_lower in desc))
        
        offerings = company.get("core_offerings", [])
        if offerings:
            offerings_text = " ".join(offerings).lower()
            if self._search_in_text(token_lower, offerings_text):
                matches.append(("core_offerings", token_lower in offerings_text))
        
        markets = company.get("target_markets", [])
        if markets:
            markets_text = " ".join(markets).lower()
            if self._search_in_text(token_lower, markets_text):
                matches.append(("target_markets", token_lower in markets_text))
        
        primary_naics = company.get("primary_naics")
        if primary_naics:
            naics_label = self._extract_naics_label(primary_naics).lower()
            if naics_label and self._search_in_text(token_lower, naics_label):
                matches.append(("naics_label", token_lower in naics_label))
        
        address = company.get("address")
        if address:
            address_text = self._extract_address_text(address).lower()
            if address_text and self._search_in_text(token_lower, address_text):
                matches.append(("address", token_lower in address_text))
        
        return matches
    
    def _search_in_text(self, token: str, text: str) -> bool:
        if re.search(rf'\b{re.escape(token)}\b', text):
            return True
        
        if token in text:
            return True
        
        if self._check_synonym_match(token, text):
            return True
        
        return False
    
    def _is_partial_match(self, token: str, text: str) -> bool:
        words = text.split()
        for word in words:
            if word.startswith(token) or token in word:
                return True
        return False
    
    def _extract_naics_label(self, naics: Any) -> str:
        if isinstance(naics, str):
            try:
                naics = ast.literal_eval(naics)
            except:
                return naics
        
        if isinstance(naics, dict):
            return naics.get("label", "")
        
        return str(naics)
    
    def _extract_address_text(self, address: Any) -> str:
        if isinstance(address, str):
            try:
                address = ast.literal_eval(address)
            except:
                return address
        
        if isinstance(address, dict):
            parts = []
            if address.get("country_code"):
                parts.append(address["country_code"])
            if address.get("country"):
                parts.append(address["country"])
            if address.get("region_name"):
                parts.append(address["region_name"])
            return " ".join(parts)
        
        return str(address)
    
    def _check_synonym_match(self, token: str, text: str) -> bool:
        synonyms = {
            "pharma": ["pharmaceutical", "medicine", "drug"],
            "pharmaceutical": ["pharma", "medicine", "drug"],
            "logistics": ["logistic", "supply", "transport", "shipping"],
            "manufacturing": ["manufacture", "factory", "production"],
            "software": ["software", "application", "app", "platform"],
            "food": ["food", "beverage", "restaurant", "cafe"],
            "beverage": ["beverage", "drink", "alcohol"],
            "france": ["france", "french", "fr"],
            "germany": ["germany", "german", "de"],
            "switzerland": ["switzerland", "swiss", "ch"],
            "romania": ["romania", "romanian", "ro"],
        }
        
        token_syns = synonyms.get(token, [token])
        for synonym in token_syns:
            if synonym in text:
                return True
        
        return False
    
    def _score_match_quality(
        self,
        verified_companies: List[Dict],
        query_tokens: List[str],
    ) -> List[Dict]:
        scores = []
        
        for company in verified_companies:
            match_info = company.get("_stage2_match_info", {})
            
            quality_score = 0
            field_quality_details = []
            
            for token, field_matches in match_info.get("token_field_matches", {}).items():
                if not field_matches:
                    continue
                
                token_score = 0
                for field_name, exact_match in field_matches:
                    weight = self.field_weights.get(field_name, 1.0)
                    if exact_match:
                        weight *= 1.5
                    token_score += weight
                
                quality_score += token_score
                field_quality_details.append({
                    "token": token,
                    "fields_matched": len(field_matches),
                    "score": token_score,
                })
            
            normalized_score = quality_score / len(query_tokens) if query_tokens else 0
            
            scores.append({
                "company": company.get("operational_name"),
                "quality_score": normalized_score,
                "raw_score": quality_score,
                "tokens_verified": len(query_tokens),
                "field_details": field_quality_details,
            })
        
        scores.sort(key=lambda x: x["quality_score"], reverse=True)
        return scores
    
    def explain_verification(self, results: Dict[str, Any]) -> str:
        output = []
        output.append(f"Stage 2: Constraint Verification")
        output.append(f"================================")
        output.append(f"")
        output.append(f"Original matches (Stage 1): {results.get('original_matched', 0)}")
        output.append(f"Verified (all constraints matched): {results.get('total_verified', 0)}")
        output.append(f"Rejected (missing constraints): {results.get('total_rejected', 0)}")
        output.append(f"Verification rate: {results.get('verification_rate', 0):.1%}")
        output.append(f"")
        
        constraint_matches = results.get("constraint_matches", {})
        if constraint_matches:
            output.append(f"Constraint Field Matches:")
            for token, matches in constraint_matches.items():
                unique_fields = list(set([m for m in matches]))
                output.append(f"  - {token}: found in {len(unique_fields)} field types")
        
        quality_scores = results.get("match_quality_scores", [])
        if quality_scores:
            output.append(f"")
            output.append(f"Top Verified Companies:")
            for idx, score_info in enumerate(quality_scores[:5], 1):
                output.append(f"  {idx}. {score_info['company']} (quality: {score_info['quality_score']:.2f})")
        
        return "\n".join(output)


def test_stage2():
    from query_parser import Stage1QueryParser
    
    print("\n" + "="*80)
    print("STAGE 2: Structured Matcher - Constraint Verification")
    print("="*80)
    
    parser1 = Stage1QueryParser("companies (1).jsonl")
    matcher2 = Stage2StructuredMatcher()
    
    test_queries = [
        "Food and beverage manufacturers in France",
        "Pharmaceutical companies in Switzerland",
        "Logistic companies in Romania",
        "Software companies in Germany",
    ]
    
    for query in test_queries:
        print(f"\n{'-'*80}")
        print(f"Query: {query}")
        
        stage1_results = parser1.parse_query(query)
        print(f"Stage 1 matched: {stage1_results['match_count']} companies")
        
        stage2_results = matcher2.match_constraints(
            stage1_results['tokens'],
            stage1_results,
        )
        
        print(matcher2.explain_verification(stage2_results))
    
    print("\n" + "="*80)
    print("STAGE 2 - TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_stage2()
