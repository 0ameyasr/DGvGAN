import os
import json

REPORTS = "sandbox/reports/raw"
API = "api.json"
PROCESSED_DIR = "sandbox/reports/processed"

# Gather directory lists once up-front
json_reports = os.listdir(REPORTS)

with open(API, "r") as f:
    labels = json.loads(f.read())

# Convert labels map for integer extraction
labels = {val: int(key) for key, val in labels.items()}

def extract():
    print("Extracting all raw behavioral reports, processing...")
    
    # Ensure processed output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed = set(os.listdir(PROCESSED_DIR))
    
    def _gather_apis(node):
        """Recursively yields any string or elements of an array found under 'api' keys."""
        if isinstance(node, dict):
            if "api" in node:
                val = node["api"]
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            yield item
                elif isinstance(val, str):
                    yield val
            
            for v in node.values():
                yield from _gather_apis(v)
                
        elif isinstance(node, list):
            for item in node:
                yield from _gather_apis(item)

    for report in json_reports:
        md5 = report[:-5]
        
        # Performance optimization: Skip reading if already processed
        if f"{md5}.json" in processed:
            print(f"Skipped sample {md5} (Already processed)")
            continue

        with open(f"{REPORTS}/{report}", "r") as f:
            data = json.load(f)
        
        # Accumulate all discovered 'api' string labels across the entire file structure
        raw_api_calls = list(_gather_apis(data))
        
        if not raw_api_calls:
            print(f"Skipped sample {md5} (No API signatures found)")
            continue
            
        # Map raw strings to numerical labels, slicing the total cumulative volume to 100 entries
        mapped_calls = [labels.get(call, None) for call in raw_api_calls][:100]
        
        sample = {
            'md5': md5,
            'api': mapped_calls
        }
        
        with open(f"{PROCESSED_DIR}/{md5}.json", "w") as f:
            json.dump(sample, f, indent=2)
            
    print("DONE.")

if __name__ == "__main__":
    extract()