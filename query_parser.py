import json
import ast
import re
import os
from typing import Dict, List, Any, Set
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Stage1QueryParser:
    
    def __init__(self, jsonl_path: str):
        self.companies = self._load_companies(jsonl_path)
        self.search_index = self._build_search_index()
        self.llm_client = self._init_llm_client()
        print(f"[OK] Loaded {len(self.companies)} companies")
        print(f"[OK] Built search index with {len(self.search_index)} searchable terms")
        if self.llm_client:
            print(f"[OK] LLM client initialized")
        
    def _init_llm_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI:
            return OpenAI(api_key=api_key)
        return None

    def _load_companies(self, jsonl_path: str) -> List[Dict[str, Any]]:
        companies = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        companies.append(json.loads(line))
        except Exception as e:
            print(f"Error loading companies: {e}")
        return companies
    
    def _build_search_index(self) -> Dict[str, Set[int]]:
        index = defaultdict(set)
        
        for idx, company in enumerate(self.companies):
            terms = self._extract_searchable_terms(company)
            
            for term in terms:
                term_lower = term.lower()
                index[term_lower].add(idx)
        
        return dict(index)
    
    def _extract_searchable_terms(self, company: Dict[str, Any]) -> Set[str]:
        terms = set()
        
        if company.get("operational_name"):
            name = company["operational_name"].lower()
            terms.add(name)
            terms.update(name.split())
        
        if company.get("description"):
            desc = company["description"].lower()
            words = re.findall(r'\b\w+\b', desc)
            terms.update(words[:50])
        
        if company.get("core_offerings"):
            offerings = company["core_offerings"]
            if isinstance(offerings, list):
                for offering in offerings:
                    terms.update(re.findall(r'\b\w+\b', offering.lower()))
        
        if company.get("target_markets"):
            markets = company["target_markets"]
            if isinstance(markets, list):
                for market in markets:
                    terms.update(re.findall(r'\b\w+\b', market.lower()))
        
        primary_naics = company.get("primary_naics")
        if primary_naics:
            if isinstance(primary_naics, str):
                try:
                    primary_naics = ast.literal_eval(primary_naics)
                except:
                    pass
            if isinstance(primary_naics, dict):
                label = primary_naics.get("label", "").lower()
                terms.update(re.findall(r'\b\w+\b', label))
        
        address = company.get("address")
        if address:
            if isinstance(address, str):
                try:
                    address = ast.literal_eval(address)
                except:
                    pass
            if isinstance(address, dict):
                country_code = address.get("country_code")
                if country_code:
                    terms.add(str(country_code).lower())
                region = address.get("region_name")
                if region:
                    terms.update(re.findall(r'\b\w+\b', str(region).lower()))
        
        return terms
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        constraints = self._extract_constraints_with_llm(query)
        
        if constraints:
            matched_indices = self._match_by_constraints(constraints)
        else:
            tokens = self._tokenize_query(query)
            matched_indices = self._match_tokens(tokens)
        
        matched_companies = [self.companies[idx] for idx in matched_indices]
        autocompleted = [self._autocomplete_company(company) for company in matched_companies]
        
        if not constraints:
            tokens = self._tokenize_query(query)
            match_scores = self._score_matches(query, matched_companies, tokens)
        else:
            match_scores = self._score_matches_by_constraints(matched_companies, constraints)
        
        return {
            "raw_query": query,
            "constraints": constraints,
            "tokens": self._tokenize_query(query),
            "matched_indices": matched_indices,
            "matched_companies": matched_companies,
            "autocompleted_companies": autocompleted,
            "match_count": len(matched_indices),
            "match_scores": match_scores,
        }
    
    def _extract_constraints_with_llm(self, query: str) -> Dict[str, Any]:
        if not self.llm_client:
            return {}
        
        try:
            prompt = f"""Extract structured constraints from this company search query.
Return a JSON object with these fields (all optional):
- industries: list of industry keywords/codes
- locations: list of countries, regions, or cities
- min_employees: minimum employee count (number)
- max_employees: maximum employee count (number)
- min_revenue: minimum revenue in dollars (number)
- max_revenue: maximum revenue in dollars (number)
- founded_after: minimum founding year (number)
- founded_before: maximum founding year (number)
- keywords: list of other important keywords
- business_type: type like "B2B", "B2C", "startup", etc.
- technologies: list of technologies or platforms

Query: "{query}"

Return only valid JSON, no additional text."""

            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            constraints = json.loads(response_text.strip())
            return constraints
        except Exception as e:
            return {}

    def _match_by_constraints(self, constraints: Dict[str, Any]) -> Set[int]:
        matched_indices = set()
        
        for idx, company in enumerate(self.companies):
            if self._company_matches_constraints(company, constraints):
                matched_indices.add(idx)
        
        if not matched_indices:
            matched_indices = set(range(len(self.companies)))
        
        return matched_indices

    def _company_matches_constraints(self, company: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        if "locations" in constraints and constraints["locations"]:
            if not self._location_matches(company, constraints["locations"]):
                return False
        
        if "industries" in constraints and constraints["industries"]:
            if not self._industry_matches(company, constraints["industries"]):
                return False
        
        if "min_employees" in constraints and constraints["min_employees"]:
            emp_count = company.get("employee_count")
            if emp_count and emp_count < constraints["min_employees"]:
                return False
        
        if "max_employees" in constraints and constraints["max_employees"]:
            emp_count = company.get("employee_count")
            if emp_count and emp_count > constraints["max_employees"]:
                return False
        
        if "min_revenue" in constraints and constraints["min_revenue"]:
            revenue = company.get("revenue")
            if revenue and revenue < constraints["min_revenue"]:
                return False
        
        if "max_revenue" in constraints and constraints["max_revenue"]:
            revenue = company.get("revenue")
            if revenue and revenue > constraints["max_revenue"]:
                return False
        
        if "founded_after" in constraints and constraints["founded_after"]:
            year = company.get("year_founded")
            if year and year < constraints["founded_after"]:
                return False
        
        if "founded_before" in constraints and constraints["founded_before"]:
            year = company.get("year_founded")
            if year and year > constraints["founded_before"]:
                return False
        
        if "keywords" in constraints and constraints["keywords"]:
            if not self._keywords_match(company, constraints["keywords"]):
                return False
        
        return True

    def _location_matches(self, company: Dict[str, Any], locations: List[str]) -> bool:
        address = company.get("address")
        if isinstance(address, str):
            try:
                address = ast.literal_eval(address)
            except:
                pass
        
        if isinstance(address, dict):
            country = address.get("country", "").lower()
            country_code = address.get("country_code", "").lower()
            region = address.get("region_name", "").lower()
            
            for loc in locations:
                loc_lower = loc.lower()
                if loc_lower in country or loc_lower in country_code or loc_lower in region:
                    return True
        
        country = company.get("country", "").lower()
        for loc in locations:
            if loc.lower() in country:
                return True
        
        return False

    def _industry_matches(self, company: Dict[str, Any], industries: List[str]) -> bool:
        naics = company.get("primary_naics")
        if isinstance(naics, str):
            try:
                naics = ast.literal_eval(naics)
            except:
                pass
        
        if isinstance(naics, dict):
            label = naics.get("label", "").lower()
            for ind in industries:
                if ind.lower() in label:
                    return True
        
        desc = company.get("description", "").lower()
        for ind in industries:
            if ind.lower() in desc:
                return True
        
        offerings = company.get("core_offerings", [])
        if offerings:
            offerings_text = " ".join(offerings).lower()
            for ind in industries:
                if ind.lower() in offerings_text:
                    return True
        
        return False

    def _keywords_match(self, company: Dict[str, Any], keywords: List[str]) -> bool:
        desc = company.get("description", "").lower()
        name = company.get("operational_name", "").lower()
        offerings = " ".join(company.get("core_offerings", [])).lower()
        markets = " ".join(company.get("target_markets", [])).lower()
        
        combined = f"{name} {desc} {offerings} {markets}".lower()
        
        for keyword in keywords:
            if keyword.lower() in combined:
                return True
        
        return False

    def _score_matches_by_constraints(self, companies: List[Dict], constraints: Dict[str, Any]) -> List[Dict]:
        scores = []
        
        for company in companies:
            score = 0
            details = []
            
            if "locations" in constraints and constraints["locations"]:
                if self._location_matches(company, constraints["locations"]):
                    score += 10
                    details.append("location_match")
            
            if "industries" in constraints and constraints["industries"]:
                if self._industry_matches(company, constraints["industries"]):
                    score += 15
                    details.append("industry_match")
            
            if "min_employees" in constraints and constraints["min_employees"]:
                emp = company.get("employee_count", 0)
                if emp >= constraints["min_employees"]:
                    score += 5
                    details.append("min_employees_ok")
            
            if "max_employees" in constraints and constraints["max_employees"]:
                emp = company.get("employee_count", float('inf'))
                if emp <= constraints["max_employees"]:
                    score += 5
                    details.append("max_employees_ok")
            
            if "min_revenue" in constraints and constraints["min_revenue"]:
                rev = company.get("revenue", 0)
                if rev >= constraints["min_revenue"]:
                    score += 5
                    details.append("min_revenue_ok")
            
            if "max_revenue" in constraints and constraints["max_revenue"]:
                rev = company.get("revenue", float('inf'))
                if rev <= constraints["max_revenue"]:
                    score += 5
                    details.append("max_revenue_ok")
            
            if "founded_after" in constraints and constraints["founded_after"]:
                year = company.get("year_founded", 0)
                if year >= constraints["founded_after"]:
                    score += 5
                    details.append("founded_after_ok")
            
            if "keywords" in constraints and constraints["keywords"]:
                if self._keywords_match(company, constraints["keywords"]):
                    score += 10
                    details.append("keywords_match")
            
            scores.append({
                "company": company.get("operational_name"),
                "score": score,
                "details": details,
            })
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    def _tokenize_query(self, query: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b', query.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'that', 'is', 'could'}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        return tokens
    
    def _match_tokens(self, tokens: List[str]) -> Set[int]:
        matched_indices = set()
        
        for token in tokens:
            if token in self.search_index:
                token_matches = self.search_index[token]
                matched_indices.update(token_matches)
            else:
                for indexed_term, indices in self.search_index.items():
                    if token in indexed_term or indexed_term.startswith(token):
                        matched_indices.update(indices)
        
        return matched_indices
    
    def _autocomplete_company(self, company: Dict[str, Any]) -> Dict[str, Any]:
        autocompleted = company.copy()
        
        if autocompleted.get("employee_count") is None:
            revenue = autocompleted.get("revenue")
            if revenue and revenue > 0:
                estimated_employees = max(1, int(revenue / 150000))
                autocompleted["employee_count"] = estimated_employees
                autocompleted["_autocompleted_employee_count"] = True
        
        if autocompleted.get("revenue") is None:
            employees = autocompleted.get("employee_count")
            if employees and employees > 0:
                estimated_revenue = employees * 150000
                autocompleted["revenue"] = estimated_revenue
                autocompleted["_autocompleted_revenue"] = True
        
        if not autocompleted.get("description"):
            core_offerings = autocompleted.get("core_offerings", [])
            target_markets = autocompleted.get("target_markets", [])
            
            if core_offerings or target_markets:
                offerings_text = ", ".join(core_offerings[:3]) if core_offerings else "various services"
                markets_text = ", ".join(target_markets[:3]) if target_markets else "general market"
                auto_desc = f"Company offering {offerings_text}, serving {markets_text}."
                autocompleted["description"] = auto_desc
                autocompleted["_autocompleted_description"] = True
        
        if autocompleted.get("year_founded") is None:
            autocompleted["year_founded"] = 2015
            autocompleted["_autocompleted_year_founded"] = True
        
        return autocompleted
    
    def _score_matches(self, query: str, companies: List[Dict], tokens: List[str]) -> List[Dict]:
        scores = []
        
        for company in companies:
            score = 0
            details = []
            
            name = (company.get("operational_name") or "").lower()
            name_matches = sum(1 for token in tokens if token in name)
            score += name_matches * 3
            if name_matches > 0:
                details.append(f"name: {name_matches}")
            
            offerings = company.get("core_offerings", [])
            offering_text = " ".join(offerings).lower()
            offering_matches = sum(1 for token in tokens if token in offering_text)
            score += offering_matches * 2
            if offering_matches > 0:
                details.append(f"offerings: {offering_matches}")
            
            desc = (company.get("description") or "").lower()
            desc_matches = sum(1 for token in tokens if token in desc)
            score += desc_matches * 1
            if desc_matches > 0:
                details.append(f"desc: {desc_matches}")
            
            scores.append({
                "company": company.get("operational_name"),
                "score": score,
                "details": details,
            })
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
    
    def explain_results(self, results: Dict[str, Any]) -> str:
        output = []
        output.append(f"Query: \"{results['raw_query']}\"")
        output.append(f"Tokens: {', '.join(results['tokens'])}")
        output.append(f"Matched companies: {results['match_count']}")
        
        if results['matched_companies']:
            output.append("\nTop Matches:")
            for idx, score_info in enumerate(results['match_scores'][:5], 1):
                output.append(f"  {idx}. {score_info['company']} (score: {score_info['score']})")
        
        return "\n".join(output)


def main():
    print("\n" + "="*80)
    print("STAGE 1 (REWRITTEN): Query Parser with Company Matching & Autocomplete")
    print("="*80)
    
    parser = Stage1QueryParser("companies (1).jsonl")
    
    test_queries = [
        "Logistic companies in Romania",
        "Pharmaceutical companies in Switzerland",
        "Software companies",
        "Companies that could supply packaging materials for cosmetics brand",
        "Manufacturing companies",
    ]
    
    for query in test_queries:
        print(f"\n{'-'*80}")
        results = parser.parse_query(query)
        
        print(parser.explain_results(results))
        
        if results['autocompleted_companies']:
            first = results['autocompleted_companies'][0]
            print(f"\nAutocomplete Example (first match):")
            print(f"  Company: {first.get('operational_name')}")
            print(f"  Original employees: {results['matched_companies'][0].get('employee_count')}")
            print(f"  Autocompleted employees: {first.get('employee_count')}")
            print(f"  Autocompleted? {'_autocompleted_employee_count' in first}")
    
    print("\n" + "="*80)
    print("STAGE 1 (REWRITTEN) - TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
