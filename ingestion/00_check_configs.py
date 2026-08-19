"""
ALWAYS run this first, before touching the dataset. Confirms the
exact config/split names so load_dataset() never throws a
guessing-game error later. Costs ~0 bandwidth.
"""
from datasets import get_dataset_config_names, get_dataset_split_names

REPO = "ai4bharat/MSMARCO-XI"

if __name__ == "__main__":
    try:
        configs = get_dataset_config_names(REPO)
        print("Available configs:", configs)
    except Exception as e:
        print("Could not fetch config names:", e)
        print("Fallback: try config='hi' directly, or 'default' if that fails.")
        configs = ["hi", "default"]

    for cfg in configs[:5]:
        try:
            splits = get_dataset_split_names(REPO, cfg)
            print(f"  config={cfg!r} -> splits={splits}")
        except Exception as e:
            print(f"  config={cfg!r} -> ERROR: {e}")

    print("\nSet backend/config.py -> hf_config to whichever config worked above.")
