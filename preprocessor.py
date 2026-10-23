import csv
import json
from pathlib import Path

def load_metadata(api_map_path, benign_evals_path, malware_evals_path):
    with open(api_map_path, "r", encoding="utf-8") as f:
        raw_api_dict = json.load(f)
    api_to_id = {api_name: int(idx) for idx, api_name in raw_api_dict.items()}
    with open(benign_evals_path, "r", encoding="utf-8") as f:
        benign_hashes = {line.strip().lower() for line in f if line.strip()}
    with open(malware_evals_path, "r", encoding="utf-8") as f:
        malware_hashes = {line.strip().lower() for line in f if line.strip()}
    return api_to_id, benign_hashes, malware_hashes

def get_parent_process(report_data):
    behavior = report_data.get("behavior", {})
    processes = behavior.get("processes", [])
    if not processes:
        return None

    proc_by_pid = {p.get("pid"): p for p in processes if "pid" in p}
    processtree = behavior.get("processtree", [])
    target = report_data.get("target", {})
    target_file = target.get("file", {}).get("name", "").lower()

    # Match process matching the target submitted filename with calls
    if target_file:
        for p in processes:
            pname = p.get("process_name", "").lower()
            if pname == target_file and len(p.get("calls", [])) > 0:
                return p

    # OR the root of report's processtree
    if processtree:
        root_pid = processtree[0].get("pid")
        proc = proc_by_pid.get(root_pid)
        if proc and len(proc.get("calls", [])) > 0:
            return proc

    # OR the process with parent_id == 0 or not found in parent PIDs
    all_pids = set(proc_by_pid.keys())
    for p in processes:
        parent_id = p.get("parent_id", 0)
        if parent_id == 0 or parent_id not in all_pids:
            if len(p.get("calls", [])) > 0:
                return p

    # OR first chronological process in the list with calls
    for p in processes:
        if len(p.get("calls", [])) > 0:
            return p

    return None

def extract_sequence(report_data, api_to_id):
    parent_proc = get_parent_process(report_data)

    if not parent_proc:
        return None, "no_parent_process"

    raw_calls = parent_proc.get("calls", [])
    if not raw_calls:
        return None, "no_calls"

    deduped_ids = []
    prev_id = None
    for call_entry in raw_calls:
        api_name = call_entry.get("api")
        if not api_name:
            continue
        if api_name not in api_to_id:
            continue
        call_id = api_to_id[api_name]

        if call_id != prev_id:
            deduped_ids.append(call_id)
            prev_id = call_id

    if len(deduped_ids) < 100:
        return None, "under_100_calls"
    return deduped_ids[:100], "success"


def export_to_csv(output_csv_path = "data/ood.csv", 
                  reports_dir = "reports/raw", 
                  api_map_path = "meta/api.json", 
                  benign_evals_path = "meta/ood_benign.txt", 
                  malware_evals_path = "meta/ood_malware.txt"):
    api_to_id, benign_hashes, malware_hashes = load_metadata(
        api_map_path, benign_evals_path, malware_evals_path
    )

    drop_reasons = {
        "corrupted_json": 0,
        "unlabeled": 0,
        "no_parent_process": 0,
        "no_calls": 0,
        "under_100_calls": 0,
    }

    raw_files = list(Path(reports_dir).glob("*.json"))
    time_headers = [f"t_{i}" for i in range(100)]
    fieldnames = ["hash"] + time_headers + ["malware"]

    written_count = 0
    counts = {"benign": 0, "malware": 0}

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for file_path in raw_files:
            md5 = file_path.stem.lower()

            if md5 in malware_hashes:
                label = 1
            elif md5 in benign_hashes:
                label = 0
            else:
                drop_reasons["unlabeled"] += 1
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
            except Exception:
                drop_reasons["corrupted_json"] += 1
                continue

            sequence, status = extract_sequence(report_data, api_to_id)
            if status != "success":
                drop_reasons[status] += 1
                continue

            writer.writerow([md5] + sequence + [label])
            written_count += 1
            if label == 1:
                counts["malware"] += 1
            else:
                counts["benign"] += 1

    print(f"Total reports analyzed: {len(raw_files)}")
    print(f"Successfully exported: {written_count}")
    print(f"*Benign: {counts['benign']}")
    print(f"*Malware: {counts['malware']}\n")
    print("Dropped:")
    for reason, qty in drop_reasons.items():
        print(f"*{reason:<24}: {qty}")

if __name__ == "__main__":
    export_to_csv()