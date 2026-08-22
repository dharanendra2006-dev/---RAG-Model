"""
Tests Tier 2 (mode: "polished") against the LIVE deployed API.
"""
import sys
import json
import time
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://rag-model-production-3de3.up.railway.app"

QUERY = "मैनहट्टन परियोजना की सफलता का प्रभाव क्या था?"

print(f"Sending polished-mode query to {BASE_URL} ...\n")

t0 = time.perf_counter()
resp = requests.post(
    f"{BASE_URL}/api/query",
    json={"text": QUERY, "mode": "polished"},
    timeout=30,
)
elapsed = (time.perf_counter() - t0) * 1000

print(f"HTTP {resp.status_code}, round-trip {elapsed:.0f}ms\n")
data = resp.json()

print(f"Status: {data.get('status')}")
print(f"\nFast answer: {(data.get('fast_answer') or {}).get('text')}")
print(f"Support score: {(data.get('fast_answer') or {}).get('support_score')}")

polished = data.get("polished_answer")
if polished:
    print(f"\nPolished text: {polished.get('text')}")
    print(f"Model used: {polished.get('model_used')}")
    print(f"Verified: {polished.get('verified')}")
    print(f"Rejection reason: {polished.get('rejection_reason')}")
else:
    print("\nNo polished_answer field in response.")

print(f"\nFull latency: {json.dumps(data.get('latency', {}), indent=2)}")
