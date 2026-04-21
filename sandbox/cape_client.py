import requests
import time

CAPE_URL = "http://localhost:8090"

def submit_file(file_path):
    files = {"file": open(file_path, "rb")}
    response = requests.post(f"{CAPE_URL}/tasks/create/file", files=files)
    return response.json()["task_id"]


def wait_for_completion(task_id, timeout=180):
    for _ in range(timeout // 5):
        res = requests.get(f"{CAPE_URL}/tasks/view/{task_id}")
        status = res.json()["task"]["status"]

        if status == "reported":
            return True

        time.sleep(5)

    return False


def get_report(task_id):
    res = requests.get(f"{CAPE_URL}/tasks/report/{task_id}")
    return res.json()