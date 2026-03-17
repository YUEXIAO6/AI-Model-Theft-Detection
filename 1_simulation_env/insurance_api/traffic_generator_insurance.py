
import requests
import time
import random
import threading
import string
import sys
import json

SERVER_URL = "http://127.0.0.1:8001"
STOP_EVENT = threading.Event()

NUM_NORMAL = 80
NUM_BATCH = 10
NUM_MONITOR = 10
NUM_S0 = 5
NUM_S1 = 5
NUM_S2 = 5
NUM_S3_THREADS = 5

S3_KEYS = [f"atk_s3_rot_{i}" for i in range(10)]

def get_padding(min_l=10, max_l=120):
    return "".join(random.choices(string.ascii_letters, k=random.randint(min_l, max_l)))

def send_req(session, url, method="GET", data=None, headers=None):
    try:
        if method == "GET": session.get(url, headers=headers, timeout=1)
        else: session.post(url, data=data, headers=headers, timeout=1)
    except Exception: pass

def get_grid_point(session):
    if not hasattr(session, "grid"):
        session.grid = [
            (a, c, y, m) 
            for a in range(20, 61, 13) 
            for c in range(10000, 50001, 13000)
            for y in range(1, 12, 5)
            for m in range(5000, 25001, 10000)
        ]
        random.shuffle(session.grid)
        session.idx = 0
    point = session.grid[session.idx % len(session.grid)]
    session.idx += 1
    return point

def make_payload_exact_len(age, car, years, mileage, target_len):
    base = {"age": age, "car_value": car, "driving_years": years, "annual_mileage": mileage, "note": ""}
    payload = json.dumps(base, separators=(",", ":"))
    base_len = len(payload.encode("utf-8"))
    if target_len <= base_len: return payload
    note_len = target_len - base_len
    for _ in range(5):
        base["note"] = "A" * max(0, note_len)
        payload = json.dumps(base, separators=(",", ":"))
        diff = len(payload.encode("utf-8")) - target_len
        if diff == 0: return payload
        note_len -= diff
    return payload

def behavior_normal(api_key, session):
    headers = {"X-API-Key": api_key, "X-Label": "Normal", "X-Type": "Legit"}
    if random.random() < 0.7:
        send_req(session, f"{SERVER_URL}/api/v1/health", headers=headers)
        time.sleep(random.uniform(0.5, 2.0))
    if random.random() < 0.3:
        payload = json.dumps({
            "age": random.randint(20, 60), "car_value": random.randint(10000, 50000),
            "driving_years": random.randint(1, 15), "annual_mileage": random.randint(5000, 25000),
            "note": get_padding()
        })
        headers["Content-Type"] = "application/json"
        send_req(session, f"{SERVER_URL}/api/v1/insurance_pricing", "POST", payload, headers)
    time.sleep(random.uniform(1.0, 5.0))

def behavior_s0(api_key, session):
    for _ in range(3):
        a, c, y, m = get_grid_point(session)
        payload = json.dumps({"age": a, "car_value": c, "driving_years": y, "annual_mileage": m, "note": get_padding()})
        headers = {"X-API-Key": api_key, "X-Label": "Attack", "X-Type": "S0_Stealthy", "Content-Type": "application/json"}
        send_req(session, f"{SERVER_URL}/api/v1/insurance_pricing", "POST", payload, headers)
        time.sleep(random.uniform(0.1, 0.3))
    time.sleep(random.uniform(300, 600))

def behavior_s1(api_key, session):
    headers = {"X-API-Key": api_key, "X-Label": "Attack", "X-Type": "S1_TimingMimic"}
    if random.random() < 0.7:
        send_req(session, f"{SERVER_URL}/api/v1/health", headers=headers)
        time.sleep(random.uniform(0.5, 2.0))
    if random.random() < 0.3:
        a, c, y, m = get_grid_point(session)
        payload = json.dumps({"age": a, "car_value": c, "driving_years": y, "annual_mileage": m, "note": get_padding()})
        headers["Content-Type"] = "application/json"
        send_req(session, f"{SERVER_URL}/api/v1/insurance_pricing", "POST", payload, headers)
    time.sleep(random.uniform(1.0, 5.0))

def behavior_s2(api_key, session):
    headers = {"X-API-Key": api_key, "X-Label": "Attack", "X-Type": "S2_SizeCamo"}
    if random.random() < 0.7:
        send_req(session, f"{SERVER_URL}/api/v1/health", headers=headers)
        time.sleep(random.uniform(0.5, 2.0))
    if random.random() < 0.3:
        a, c, y, m = get_grid_point(session)
        target_len = random.randint(120, 220) 
        payload = make_payload_exact_len(a, c, y, m, target_len)
        headers["Content-Type"] = "application/json"
        send_req(session, f"{SERVER_URL}/api/v1/insurance_pricing", "POST", payload, headers)
    time.sleep(random.uniform(1.0, 5.0))

def user_s3_rotation():
    session = requests.Session()
    while not STOP_EVENT.is_set():
        current_key = random.choice(S3_KEYS)
        if random.random() < 0.95:
            behavior_normal(current_key, session)
        else:
            a, c, y, m = get_grid_point(session)
            payload = json.dumps({"age": a, "car_value": c, "driving_years": y, "annual_mileage": m, "note": get_padding()})
            headers = {"X-API-Key": current_key, "X-Label": "Attack", "X-Type": "S3_KeyRotation", "Content-Type": "application/json"}
            send_req(session, f"{SERVER_URL}/api/v1/insurance_pricing", "POST", payload, headers)
            time.sleep(random.uniform(1.0, 5.0))

def behavior_batch(api_key, session):
    headers = {"X-API-Key": api_key, "X-Label": "Normal", "X-Type": "LegitBatch", "Content-Type": "application/json"}
    for i in range(random.randint(5, 10)):
        if STOP_EVENT.is_set(): break
        payload = json.dumps({"age": 30, "car_value": 20000+i*1000, "driving_years": 5, "annual_mileage": 15000, "note": f"Batch {i}"})
        send_req(session, f"{SERVER_URL}/api/v1/insurance_pricing", "POST", payload, headers)
        time.sleep(random.uniform(0.3, 0.8))
    time.sleep(random.uniform(10, 30))

def behavior_monitor(api_key, session):
    headers = {"X-API-Key": api_key, "X-Label": "Normal", "X-Type": "Monitoring"}
    for _ in range(random.randint(3, 7)):
        if STOP_EVENT.is_set(): break
        send_req(session, f"{SERVER_URL}/api/v1/health", headers=headers)
        time.sleep(random.uniform(0.5, 1.2))
    time.sleep(random.uniform(5, 15))

def user_lifecycle(user_type, api_key):
    session = requests.Session()
    while not STOP_EVENT.is_set():
        if user_type == "Normal": behavior_normal(api_key, session)
        elif user_type == "Batch": behavior_batch(api_key, session)
        elif user_type == "Monitor": behavior_monitor(api_key, session)
        elif user_type == "S0": behavior_s0(api_key, session)
        elif user_type == "S1": behavior_s1(api_key, session)
        elif user_type == "S2": behavior_s2(api_key, session)

if __name__ == "__main__":
    threads = []
    for i in range(NUM_NORMAL): threads.append(threading.Thread(target=user_lifecycle, args=("Normal", f"norm_{i}"), daemon=True))
    for i in range(NUM_BATCH): threads.append(threading.Thread(target=user_lifecycle, args=("Batch", f"batch_{i}"), daemon=True))
    for i in range(NUM_MONITOR): threads.append(threading.Thread(target=user_lifecycle, args=("Monitor", f"monitor_{i}"), daemon=True))
    for i in range(NUM_S0): threads.append(threading.Thread(target=user_lifecycle, args=("S0", f"atk_s0_{i}"), daemon=True))
    for i in range(NUM_S1): threads.append(threading.Thread(target=user_lifecycle, args=("S1", f"atk_s1_{i}"), daemon=True))
    for i in range(NUM_S2): threads.append(threading.Thread(target=user_lifecycle, args=("S2", f"atk_s2_{i}"), daemon=True))
    for _ in range(NUM_S3_THREADS): threads.append(threading.Thread(target=user_s3_rotation, daemon=True))

    for t in threads: t.start()
    print("traffic_generator_insurance.py running... ")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        STOP_EVENT.set()
        sys.exit(0)