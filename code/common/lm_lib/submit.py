"""Write prediction.json in the Phase-1 format and zip it with the file at the archive root."""
import json
import os
import zipfile


def write_submission(preds, test_prompts, out_dir, name="submission"):
    """preds: {id: continuation}. test_prompts: the official test dict (used to validate IDs)."""
    expected, got = set(test_prompts), set(preds)
    missing, extra = expected - got, got - expected
    if missing or extra:
        raise ValueError(f"ID mismatch: {len(missing)} missing (e.g. {sorted(missing)[:5]}), "
                         f"{len(extra)} unknown (e.g. {sorted(extra)[:5]})")
    empty = [i for i, t in preds.items() if not t.strip()]
    if empty:
        print(f"WARNING: {len(empty)} empty continuations, e.g. {empty[:5]}")
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump({i: {"text": preds[i]} for i in test_prompts}, f, ensure_ascii=False, indent=1)
    zip_path = os.path.join(out_dir, f"{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
    print(f"wrote {pred_path} and {zip_path} ({len(preds)} predictions)")
    return zip_path
