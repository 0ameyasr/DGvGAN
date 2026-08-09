import time
import os
import json
import datetime
import requests
import pandas

from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = "meta/vt_cache_benign.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_to_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)


def get_vt_file_details_safe(md5_hash, api_key, cache):
    if md5_hash in cache:
        print(f"[+] {md5_hash} found in local cache, skipping.")
        return cache[md5_hash], False

    url = f"https://www.virustotal.com/api/v3/files/{md5_hash}"
    headers = {"accept": "application/json", "x-apikey": api_key}

    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            print("[-] Quota exceeded, waiting 60 seconds before retrying...")
            time.sleep(60)
            return get_vt_file_details_safe(md5_hash, api_key, cache)

        if response.status_code == 404:
            print(f"[-] Hash {md5_hash} not found in VirusTotal.")
            return None, True

        response.raise_for_status()

        attributes = response.json().get("data", {}).get("attributes", {})

        def format_epoch(epoch):
            return (
                datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                if epoch
                else "N/A"
            )

        details = {
            "Internal Creation Date": format_epoch(attributes.get("creation_date")),
            "First Seen on VirusTotal": format_epoch(
                attributes.get("first_submission_date")
            ),
            "Malicious Detections": attributes.get("last_analysis_stats", {}).get(
                "malicious", 0
            ),
        }

        return details, True

    except requests.exceptions.RequestException as e:
        print(f"[-] API Error: {e}")
        return None, False


if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY")

    # with open("meta/malware_evals.txt","r") as f:
    #     hash_list = [h for h in f.read().split('\n') if h]

    with open("meta/benign_evals.txt", "r") as f:
        hash_list = [h for h in f.read().split("\n") if h]

    cache = load_cache()

    for index, md5 in enumerate(hash_list):
        info, api_used = get_vt_file_details_safe(md5, API_KEY, cache)

        if info and api_used:
            cache[md5] = info
            print()
            print(info)
            print()
            save_to_cache(cache)

            if index < len(hash_list) - 1:
                print("[~] Sleeping 16 seconds to protect rate limit...")
                time.sleep(16)
