import time
import os
import json
import datetime
import requests

CACHE_FILE = "vt_cache.json"

def load_cache():
    """Loads previously fetched results from a local JSON file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_to_cache(cache_data):
    """Saves lookup results locally to prevent duplicate API calls."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)

def get_vt_file_details_safe(md5_hash, api_key, cache):
    if md5_hash in cache:
        print(f"[+] {md5_hash} found in local cache. No API call needed.")
        return cache[md5_hash], False  
    
    url = f"https://www.virustotal.com/api/v3/files/{md5_hash}"
    headers = {"accept": "application/json", "x-apikey": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            print("[-] Quota exceeded! Waiting 60 seconds before retrying...")
            time.sleep(60)
            return get_vt_file_details_safe(md5_hash, api_key, cache)
            
        if response.status_code == 404:
            print(f"[-] Hash {md5_hash} not found in VirusTotal.")
            return None, True
            
        response.raise_for_status()
        
        attributes = response.json().get('data', {}).get('attributes', {})
        def format_epoch(epoch):
            return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if epoch else "N/A"
            
        details = {
            "Internal Creation Date": format_epoch(attributes.get("creation_date")),
            "First Seen on VirusTotal": format_epoch(attributes.get("first_submission_date")),
            "Malicious Detections": attributes.get("last_analysis_stats", {}).get("malicious", 0)
        }
        
        return details, True  
    
    except requests.exceptions.RequestException as e:
        print(f"[-] API Error: {e}")
        return None, False

if __name__ == "__main__":
    import os, pandas
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.getenv("API_KEY")
    
    hash_list_df = pandas.read_csv("evaluated.csv")
    hash_list_df['sample'] = hash_list_df['sample'].str.removesuffix('.json')
    hash_list = hash_list_df['sample']
    
    cache = load_cache()
    
    for index, md5 in enumerate(hash_list):
        print(f"\nProcessing [{index + 1}/{len(hash_list)}]: {md5}")
        
        info, api_used = get_vt_file_details_safe(md5, API_KEY, cache)
        
        if info and api_used:
            cache[md5] = info
            print()
            print(info)
            print()
            save_to_cache(cache)
            
            if index < len(hash_list) - 1:
                print("[~] Sleeping 16 seconds to protect Public API rate limit...")
                time.sleep(16)
        elif info:
            print(f"    Creation Date: {info['Internal Creation Date']}")