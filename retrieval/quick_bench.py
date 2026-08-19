import sys
sys.path.insert(0, ".")
from hybrid_search import hybrid_retrieve

queries = [
    "मैनहट्टन परियोजना की सफलता का प्रभाव क्या था?",
    "भारत की राजधानी क्या है?",
    "द्वितीय विश्व युद्ध कब समाप्त हुआ?",
    "परमाणु बम का आविष्कार किसने किया?",
    "मैनहट्टन परियोजना कब शुरू हुई?",
]

for q in queries:
    out = hybrid_retrieve(q)
    lat = out["latency"]
    total = lat["query_embed_and_dense_ms"] + lat["sparse_ms"] + lat["fusion_ms"]
    print(f"{total:.1f}ms  (embed+dense={lat['query_embed_and_dense_ms']:.1f}, sparse={lat['sparse_ms']:.1f}, fusion={lat['fusion_ms']:.3f})")
