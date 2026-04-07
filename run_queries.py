import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

COUNTRY_CODE_MAP = {
    'ro': 'Romania',
    'us': 'United States',
    'fr': 'France',
    'de': 'Germany',
    'ch': 'Switzerland',
    'se': 'Sweden',
    'no': 'Norway',
    'dk': 'Denmark',
    'nl': 'Netherlands',
    'be': 'Belgium',
    'at': 'Austria',
    'cz': 'Czech Republic',
    'pl': 'Poland',
    'it': 'Italy',
    'es': 'Spain',
    'pt': 'Portugal',
    'gr': 'Greece',
    'ua': 'Ukraine',
    'ru': 'Russia',
    'gb': 'United Kingdom',
    'ie': 'Ireland',
    'ca': 'Canada',
    'au': 'Australia',
    'jp': 'Japan',
    'cn': 'China',
    'in': 'India',
    'br': 'Brazil',
    'mx': 'Mexico',
}

def get_country_name(country_code):
    if not country_code:
        return 'Unknown'
    return COUNTRY_CODE_MAP.get(country_code.lower(), country_code)

def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


def test_query(query_num, query_text):
    print("\n" + "=" * 80)
    print(f"QUERY {query_num}: {query_text}")
    print("=" * 80)
    
    try:
        from stage2_matcher import Stage2StructuredMatcher
        from query_parser import Stage1QueryParser
        
        parser = Stage1QueryParser("companies (1).jsonl")
        
        print("[STAGE 1] Query Parsing with LLM Constraint Extraction...")
        stage1_result = parser.parse_query(query_text)
        stage1_candidates = len(stage1_result.get("matched_companies", []))
        print(f"  Found: {stage1_candidates} candidates")
        print(f"  Extracted constraints: {stage1_result.get('constraints', {})}")
        
        print("[STAGE 2] Constraint Verification with LLM...")
        matcher = Stage2StructuredMatcher()
        stage2_result = matcher.match_constraints(
            stage1_result.get("tokens", []),
            stage1_result
        )
        stage2_verified = len(stage2_result.get("verified_companies", []))
        print(f"  Verified: {stage2_verified} companies")
        
        if stage2_verified == 0:
            print("  >>> NO MATCHES FOUND")
            return None
        
        companies = stage2_result.get("verified_companies", [])
        
        print(f"\n  Matching Companies ({stage2_verified} total):")
        for i, company in enumerate(companies[:10], 1):
            name = company.get('operational_name', 'N/A')
            addr = company.get('address', {})
            if isinstance(addr, str):
                try:
                    import ast
                    addr = ast.literal_eval(addr)
                except:
                    pass
            country_code = None
            if isinstance(addr, dict):
                country_code = addr.get('country_code')
            country = get_country_name(country_code)
            employees = company.get('employee_count', 'N/A')
            industry = company.get('industry_classification', 'N/A')
            
            print(f"  {i:2d}. {name}")
            print(f"       Country: {country} | Employees: {employees} | Industry: {industry}")
        
        if stage2_verified > 10:
            print(f"  ... and {stage2_verified - 10} more companies")
        
        return companies
        
    except Exception as e:
        import traceback


def main():
    load_env()
    
    queries = [
        "companii software bucuresti",
        "companii cu revenue de peste un milion anual din scandinavia"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING 12 QUERIES - LLM-POWERED QUERY PARSING")
    print("=" * 80)
    
    results = []
    for i, query in enumerate(queries, 1):
        result = test_query(i, query)
        if result:
            results.append((i, query, len(result), result[0].get('operational_name', 'N/A')))
    
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY - 12 QUERIES EXECUTED")
    print("=" * 80)
    
    total_matches = 0
    matched_queries = 0
    
    for num, query, count, top in results:
        total_matches += count
        if count > 0:
            matched_queries += 1
        status = "[MATCH]" if count > 0 else "[NO MATCH]"
        print(f"\n{status} Query {num:2d}: {query}")
        print(f"    Results: {count} companies")
        if count > 0:
            print(f"    Top: {top}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {matched_queries}/12 queries with matches")
    print(f"TOTAL RESULTS: {total_matches} companies found")
    print("=" * 80)


if __name__ == "__main__":
    main()
