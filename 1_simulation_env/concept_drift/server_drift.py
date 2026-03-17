# server_drift.py
import uvicorn
from fastapi import FastAPI, Request
import time
import random
import os
import hashlib
import asyncio
import json
import math
import csv
import threading

LOG_FILE = "traffic_logs_drift.csv"
NORMAL_SUBNETS = ["192.168.1.", "10.0.5.", "172.16.10.", "203.0.113."]
FILE_LOCK = threading.Lock()

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "src_ip", "api_key", "endpoint", "method", "proc_duration",
            "fwd_bytes", "bwd_bytes", "status_code", "label", "type",
            "age", "income", "body_len"
        ])
app = FastAPI()

def get_consistent_ip(api_key: str) -> str:
    if (not api_key) or ("unknown" in api_key):
        return "192.168.1.50"
    hash_val = int(hashlib.md5(api_key.encode()).hexdigest(), 16)
    subnet = NORMAL_SUBNETS[hash_val % len(NORMAL_SUBNETS)]
    return f"{subnet}{(hash_val % 253) + 2}"

@app.middleware("http")
async def log_traffic(request: Request, call_next):
    arrival_time = time.time()
    api_key = request.headers.get("X-API-Key", "unknown")
    label = request.headers.get("X-Label", "Normal")
    traffic_type = request.headers.get("X-Type", "Legit")
    
    src_ip = request.headers.get("X-Fake-IP", get_consistent_ip(api_key))
    
    body_bytes = await request.body()
    body_len = len(body_bytes)
    age_val, income_val = "", ""
    if body_bytes:
        try:
            body_json = json.loads(body_bytes)
            age_val = body_json.get("age", "")
            income_val = body_json.get("income", "")
        except Exception:
            pass
            
    EXCLUDE_HEADERS = {"x-api-key", "x-label", "x-fake-ip", "x-type", "host", "user-agent", "content-length"}
    header_bytes = sum(len(k) + len(v) for k, v in request.headers.items() if k.lower() not in EXCLUDE_HEADERS)
    fwd_bytes = 54 + header_bytes + body_len
    
    start_proc = time.perf_counter()
    response = await call_next(request)
    proc_duration = time.perf_counter() - start_proc
    
    content_len = response.headers.get("content-length")
    bwd_bytes = int(content_len) + 54 if content_len else 500
    
    row = [
        f"{arrival_time:.6f}", src_ip, api_key, request.url.path, request.method,
        f"{proc_duration:.6f}", int(fwd_bytes), int(bwd_bytes), int(response.status_code),
        label, traffic_type, age_val, income_val, int(body_len)
    ]
    with FILE_LOCK:
        with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
    return response

@app.post("/api/v1/credit_score")
async def get_credit_score(request: Request):
    await asyncio.sleep(random.uniform(0.05, 0.3))
    body = await request.json()
    age, income = float(body.get("age", 30)), float(body.get("income", 50000))
    score = 0.4 * math.sin((age / 100.0) * math.pi) + 0.6 * (1 - math.exp(-(income / 150000.0) * 3))
    return {"risk_score": round(max(0.0, min(1.0, score)), 4)}

@app.get("/api/v1/health")
async def health_check():
    await asyncio.sleep(random.uniform(0.05, 0.3))
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)