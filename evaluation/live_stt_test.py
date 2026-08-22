"""
Sends a real recorded audio file to the LIVE deployed API's
/api/query endpoint, base64-encoded, exercising the full voice path
end to end on the actual deployment.
"""
import sys
import base64
import json
import time
import requests

BASE_URL = sys.argv[2] if len(sys.argv) > 2 else "https://rag-model-production-3de3.up.railway.app"
AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else None

CONTENT_TYPES = {
    "m4a": "audio/mp4",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "webm": "audio/webm",
}


def main():
    if not AUDIO_PATH:
        print("Usage: python live_stt_test.py <path_to_audio_file> [base_url]")
        sys.exit(1)

    with open(AUDIO_PATH, "rb") as f:
        audio_bytes = f.read()

    ext = AUDIO_PATH.rsplit(".", 1)[-1].lower()
    content_type = CONTENT_TYPES.get(ext, "audio/webm")

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    print(f"Sending {len(audio_bytes)} bytes ({ext}, {content_type}) to {BASE_URL}/api/query ...")

    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/api/query",
        json={"audio_base64": audio_b64, "audio_content_type": content_type, "mode": "fast"},
        timeout=30,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\nHTTP {resp.status_code}, round-trip {elapsed:.0f}ms\n")
    data = resp.json()
    print(f"Status: {data.get('status')}")
    print(f"Transcript: {data.get('transcript')}")
    if data.get("fast_answer"):
        print(f"Answer: {data['fast_answer']['text']}")
        print(f"Support score: {data['fast_answer']['support_score']}")
    if data.get("message"):
        print(f"Message: {data['message']}")
    print(f"\nLatency breakdown: {json.dumps(data.get('latency', {}), indent=2)}")


if __name__ == "__main__":
    main()
