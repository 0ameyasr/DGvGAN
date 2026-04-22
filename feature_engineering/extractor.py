import json

def extract_features_from_report(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processes = data.get("behavior", {}).get("processes", [])
    files = data.get("behavior", {}).get("summary", {}).get("files", [])
    domains = data.get("network", {}).get("domains", [])
    ips = data.get("network", {}).get("ips", [])
    http = data.get("network", {}).get("http_requests", [])

    # Safe filtering (IMPORTANT FIX)
    valid_processes = [p for p in processes if isinstance(p, dict)]

    num_processes = len(valid_processes)
    num_files = len(files)
    num_ips = len(ips)

    suspicious_files = sum(1 for f in files if isinstance(f, str) and "encrypted" in f.lower())

    exe_processes = sum(
        1 for p in valid_processes
        if "process_name" in p and isinstance(p["process_name"], str) and ".exe" in p["process_name"].lower()
    )

    return [num_processes, num_files, num_ips]