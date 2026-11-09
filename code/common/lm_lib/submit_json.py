"""Generic Codabench writer for Phases 2-7: prediction.json at the root of submission.zip, with ID checks."""
import json
import os
import zipfile


def write_json_submission(obj, expected_ids, out_dir, name="submission"):
    got = set(obj) if isinstance(obj, dict) else {str(x["id"]) for x in obj}
    expected = {str(i) for i in expected_ids}
    missing, extra = expected - got, got - expected
    if missing or extra:
        raise ValueError(f"ID mismatch: {len(missing)} missing (e.g. {sorted(missing)[:5]}), "
                         f"{len(extra)} unknown (e.g. {sorted(extra)[:5]})")
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    zip_path = os.path.join(out_dir, f"{name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"], z.namelist()
    print(f"wrote {zip_path} ({len(got)} ids)")
    return zip_path
