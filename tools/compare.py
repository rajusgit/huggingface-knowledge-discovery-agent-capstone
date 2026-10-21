"""Side-by-side comparison of models/datasets with explicit missing-field warnings and
user-constraint checks. Facts come only from live Hub metadata; nothing is inferred."""
from tools import hf_client

FIELDS = ["pipeline_tag", "library_name", "license", "parameters", "languages",
          "downloads", "likes", "revision", "gated"]
WARN_FIELDS = ["license", "parameters", "languages", "pipeline_tag"]
CONSTRAINTS = {"licenses", "max_parameters", "language", "task"}
MIN_IDS, MAX_IDS = 2, 5


def _check(row: dict, c: dict) -> dict:
    violations, unknown = [], []
    if "licenses" in c:
        if row["license"] is None:
            unknown.append("licenses")
        elif row["license"].lower() not in {x.lower() for x in c["licenses"]}:
            violations.append(f"license {row['license']} not in allowed {sorted(c['licenses'])}")
    if "max_parameters" in c:
        if row["parameters"] is None:
            unknown.append("max_parameters")
        elif row["parameters"] > c["max_parameters"]:
            violations.append(f"parameters {row['parameters']:,} exceeds {c['max_parameters']:,}")
    if "language" in c:
        if row["languages"] is None:
            unknown.append("language")
        elif c["language"].lower() not in {l.lower() for l in row["languages"]}:
            violations.append(f"language {c['language']} not supported (has {row['languages']})")
    if "task" in c:
        if row["pipeline_tag"] is None:
            unknown.append("task")
        elif row["pipeline_tag"] != c["task"]:
            violations.append(f"task is {row['pipeline_tag']}, wanted {c['task']}")
    return {"ok": not violations, "violations": violations, "unknown": unknown}


def compare_resources(ids: list[str], kind: str = "models", constraints: dict | None = None) -> dict:
    ids = list(ids or [])
    if len(set(ids)) != len(ids) or not MIN_IDS <= len(ids) <= MAX_IDS:
        raise ValueError(f"provide {MIN_IDS}-{MAX_IDS} distinct ids")
    if kind not in ("models", "datasets"):
        raise ValueError("kind must be 'models' or 'datasets'")
    constraints = constraints or {}
    extra = set(constraints) - CONSTRAINTS
    if extra:
        raise ValueError(f"unknown constraint(s) {sorted(extra)}; allowed: {sorted(CONSTRAINTS)}")
    fetch = hf_client.get_model_details if kind == "models" else hf_client.get_dataset_details
    details = [fetch(i) for i in ids]
    rows = [{"id": d["id"], **{f: d.get(f) for f in FIELDS}, "source_url": d["source_url"],
             "fetched_at": d["fetched_at"]} for d in details]
    warnings = [{"id": r["id"], "field": f, "message": f"{f} not stated by the Hub for {r['id']}"}
                for r in rows for f in WARN_FIELDS if r[f] is None]
    out = {
        "models": rows,
        "fields": FIELDS,
        "warnings": warnings,
        "fetched_at": max(r["fetched_at"] for r in rows),
        "time_sensitive_fields": hf_client.TIME_SENSITIVE,
    }
    if constraints:
        out["constraints"] = constraints
        out["constraints_check"] = {r["id"]: _check(r, constraints) for r in rows}
    return out
