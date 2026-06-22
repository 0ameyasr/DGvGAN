import os
import json

raw_reports = set(os.listdir("sandbox/reports/raw"))
proc_reports = set(os.listdir("sandbox/reports/processed"))

no_api = []
for proc_report in proc_reports:
    with open(f"sandbox/reports/processed/{proc_report}","r") as f:
        report_json = json.load(f)
        api = report_json['api'] if report_json else None
        
        if (not api) or (None in api):
            no_api.append(proc_report)
                    
print(f"A total of {len(no_api)}/{len(proc_reports)} reports have not been parsed correctly or may not include api data.")

print()
# unprocessed_raw = ""
print("Following have not been processed from raw:")
for raw_report in raw_reports:
    if raw_report not in proc_reports:
        with open(f"sandbox/reports/raw/{raw_report}","r") as f:
            report_json = json.load(f)
            # unprocessed_raw += f"{str(report_json)}\n"

# with open("unproc_reports.txt","w",encoding='utf-8') as f:
#     f.write(str(unprocessed_raw))