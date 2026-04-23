import os
import json

REPORTS = "reports/raw"
API = "api.json"

json_reports = os.listdir(REPORTS)

with open(API,"r") as f:
    labels = json.loads(f.read())

labels = {val:int(key) for key, val in labels.items()}

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
    processes = data['behavior']['processes']
    md5 = report[:-5]
    
    print(f"MD5: {md5}")
    sample = {}
    for process in processes:
        api_calls = resolve(process)
        if api_calls:
            sample['md5'] = md5
            sample['api'] = [labels.get(call,None) for call in api_calls][:100]
            
    with open(f"reports/processed/{md5}.json","w") as f:
        json.dump(sample,f,indent=2)