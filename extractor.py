import os
import json

REPORTS = "sandbox/reports/raw"
API = "api.json"

json_reports = os.listdir(REPORTS)

with open(API,"r") as f:
    labels = json.loads(f.read())

labels = {val:int(key) for key, val in labels.items()}

def extract():
    print("Extracting all raw behavioral reports, processing...")
    def resolve(process: dict):
        process_calls = process.get('calls', None)
        
        api_calls = []
        if process_calls:
            for call in process_calls:
                api_call = call['api']
                api_calls.append(api_call)
        
        return api_calls
        
    for report in json_reports:
        with open(f"{REPORTS}/{report}", "r") as f:
            data = json.load(f)

        metadata = data['metadata']['output']['pcap']
        md5 = report[:-5]
        
        if 'behavior' not in data:
            print(f"Skipped sample {md5}")
            continue
        
        processes = data['behavior']['processes']
        
        sample = {}
        for process in processes:
            api_calls = resolve(process)
            if api_calls:
                sample['md5'] = md5
                sample['api'] = [labels.get(call,None) for call in api_calls][:100]
                
        processed = set(os.listdir("sandbox/reports/processed"))
        if f"{md5}.json" in processed:
            print(f"Skipped sample {md5}")
            continue
        else:
            with open(f"sandbox/reports/processed/{md5}.json","w") as f:
                json.dump(sample,f,indent=2)
    print("DONE.")