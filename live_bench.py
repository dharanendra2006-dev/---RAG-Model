import sys, time, json
import urllib.request

QUERIES = [
    "मैनहट्टन परियोजना की सफलता का प्रभाव क्या था?",
    "परमाणु बम किसने बनाया?",
    "मैनहट्टन परियोजना कब शुरू हुई?",
    "द्वितीय विश्व युद्ध कब समाप्त हुआ?",
    "परमाणु ऊर्जा का शांतिपूर्ण उपयोग क्या है?",
]

def query(text):
    body = json.dumps({"text": text, "mode": "fast"}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/api/query",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

latencies = []
for i in range(20):
    q = QUERIES[i % len(QUERIES)]
    result = query(q)
    lat = result["latency"]["fast_path_total_ms"]
    latencies.append(lat)
    print(f"{i+1}: {lat:.2f}ms")

latencies.sort()
n = len(latencies)
p50 = latencies[int(n*0.5)]
p70 = latencies[int(n*0.7)]
p100 = latencies[-1]
print(f"\nP50={p50:.2f}ms P70={p70:.2f}ms P100={p100:.2f}ms")
print(f"Under 50ms: {sum(1 for l in latencies if l < 50)}/{n}")
print(f"Under 200ms: {sum(1 for l in latencies if l < 200)}/{n}")
