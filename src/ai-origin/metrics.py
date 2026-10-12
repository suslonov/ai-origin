"""Selection and scoring without an LLM: deterministic, inspectable arithmetic.

Agreement with manual judgments and accuracy against known authorship answer
different questions. They are intentionally reported as separate metric groups.
All errors/biases are percentage points; rates and Brier score are 0–1 values.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from schemas import RunInput


def select_samples(samples: list[dict], config: RunInput) -> list[dict]:
    """Balanced random sampling uses equal turns across language × score band.

    Sort first so filesystem order cannot change a seeded selection. Shuffle
    within each stratum and the strata themselves; then draw round-robin until
    ceil(percent × eligible count) distinct samples have been selected. Small
    groups can run out: the remaining groups fill the unused places.
    """
    eligible = sorted((s for s in samples if config.split == "all" or s["split"] == config.split), key=lambda s: s["id"])
    if config.strategy == "manual":
        ids = set(config.sample_ids)
        found = {s["id"] for s in eligible}
        if not ids or ids - found:
            raise ValueError("Select existing samples within the selected dataset split.")
        return [s for s in eligible if s["id"] in ids]
    if config.strategy == "all":
        return eligible
    count = min(len(eligible), math.ceil(len(eligible) * config.percent / 100))
    rng = random.Random(config.seed)
    groups: dict[tuple, list] = defaultdict(list)
    for sample in eligible:
        band = min(2, int(sample["manual_probability"] / (100 / 3)))
        groups[(sample["lang"], band)].append(sample)
    buckets = list(groups.values())
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    selected = []
    while len(selected) < count:
        for bucket in buckets:
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
    return selected


def _ratio(num: float, den: int) -> float | None:
    return num / den if den else None


def quality_metrics(rows: list[dict], threshold: float) -> dict[str, Any]:
    """Failed/pending responses never count as zero-probability predictions.

    No denominator means None (shown as a dash), not 0% accuracy. The acceptable
    interval score penalizes only the distance outside the user's labeled range.
    Confirmed human/AI labels alone enter the confusion matrix and Brier score.
    """
    good = [r for r in rows if r["status"] == "completed"]
    errors = [r["model_probability"] - r["sample"]["manual_probability"] for r in good]
    distances = [max(r["sample"]["manual_low"] - r["model_probability"], 0,
                     r["model_probability"] - r["sample"]["manual_high"]) for r in good]
    known = [r for r in good if r["sample"]["origin"] in {"human", "ai"}]
    tp = sum(r["sample"]["origin"] == "ai" and r["model_probability"] >= threshold for r in known)
    fn = sum(r["sample"]["origin"] == "ai" and r["model_probability"] < threshold for r in known)
    fp = sum(r["sample"]["origin"] == "human" and r["model_probability"] >= threshold for r in known)
    tn = sum(r["sample"]["origin"] == "human" and r["model_probability"] < threshold for r in known)
    n = len(good)
    return {
        "total": len(rows), "completed": n, "failed": sum(r["status"] == "failed" for r in rows),
        "pending": sum(r["status"] == "pending" for r in rows),
        "mae_pp": _ratio(sum(abs(e) for e in errors), n),
        "rmse_pp": math.sqrt(sum(e * e for e in errors) / n) if n else None,
        "bias_pp": _ratio(sum(errors), n),
        "within_range": _ratio(sum(d == 0 for d in distances), n),
        "range_distance_pp": _ratio(sum(distances), n),
        "known_origin": {"count": len(known), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                         "threshold": threshold, "accuracy": _ratio(tp + tn, len(known)),
                         "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
                         "false_positive_rate": _ratio(fp, fp + tn),
                         "brier": _ratio(sum((r["model_probability"] / 100 - int(r["sample"]["origin"] == "ai")) ** 2 for r in known), len(known))},
    }


def grouped_metrics(rows: list[dict], threshold: float) -> dict:
    result = {"overall": quality_metrics(rows, threshold)}
    for field in ("lang", "materials_mode", "split"):
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            groups[row["sample"][field]].append(row)
        result[field] = {key: quality_metrics(group, threshold) for key, group in sorted(groups.items())}
    return result


def comparison(rows: list[dict], reference: dict) -> dict:
    """Compare only pairs with successful outputs on both sides.

    A negative MAE delta means the candidate got closer to manual labels.
    This is agreement, not a claim of better real-world authorship detection.
    """
    old = {r["sample"]["id"]: r for r in reference["results"] if r["status"] == "completed"}
    pairs = [(r, old[r["sample"]["id"]]) for r in rows if r["status"] == "completed" and r["sample"]["id"] in old]
    deltas = [abs(r["model_probability"] - r["sample"]["manual_probability"]) -
              abs(b["model_probability"] - b["sample"]["manual_probability"]) for r, b in pairs]
    return {"reference_run_id": reference["id"], "paired_samples": len(pairs),
            "mae_change_pp": _ratio(sum(deltas), len(deltas)),
            "improved": sum(d < -1e-8 for d in deltas), "regressed": sum(d > 1e-8 for d in deltas),
            "unchanged": sum(abs(d) <= 1e-8 for d in deltas)}
