from collections import defaultdict

import numpy as np


def clustered_interval(values, groups, samples=1000, seed=42, denominators=None):
    if len(values) != len(groups) or not len(values):
        raise ValueError("Nonempty matching values and groups required")
    values = np.asarray(values, dtype=float)
    denominator = (
        np.ones(len(values)) if denominators is None else np.asarray(denominators, dtype=float)
    )
    if (
        denominator.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(denominator).all()
        or (denominator < 0).any()
        or denominator.sum() <= 0
    ):
        raise ValueError("Invalid values or denominators")
    indices = {g: i for i, g in enumerate(sorted(set(groups)))}
    positions = [indices[g] for g in groups]
    totals = np.bincount(positions, weights=values, minlength=len(indices))
    counts = np.bincount(positions, weights=denominator, minlength=len(indices))
    value = float(totals.sum() / counts.sum())
    result = {"value": value, "n": len(values), "n_groups": len(indices), "ci95": None}
    if len(indices) < 2:
        return result
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selection = rng.integers(len(indices), size=len(indices))
        denominator = counts[selection].sum()
        if denominator > 0:
            estimates.append(totals[selection].sum() / denominator)
    if estimates:
        result["ci95"] = np.quantile(estimates, [0.025, 0.975]).tolist()
    return result


def summarize(rows, samples=1000, seed=42):
    keys = sorted({k for row in rows for k in row["metrics"]})
    result = {}
    for key in keys:
        selected = [r for r in rows if r["metrics"].get(key) is not None]
        if selected:
            result[key] = clustered_interval(
                [r["metrics"][key] for r in selected],
                [r["semantic_group"] for r in selected],
                samples,
                seed,
            )
            result[key].update(
                coverage=len(selected) / len(rows), missing=len(rows) - len(selected)
            )
            if key.endswith("_seconds") or key in {"real_time_factor", "peak_memory_bytes"}:
                result[key]["quantiles"] = dict(
                    zip(
                        ["p50", "p95", "p99"],
                        np.quantile(
                            [r["metrics"][key] for r in selected], [0.5, 0.95, 0.99]
                        ).tolist(),
                        strict=True,
                    )
                )
        else:
            result[key] = {
                "value": None,
                "n": 0,
                "n_groups": 0,
                "ci95": None,
                "coverage": 0.0,
                "missing": len(rows),
            }
    for prefix in ["text_answer", "spoken_answer", "tts", "input_asr"]:
        for unit, suffix in [("words", "wer"), ("characters", "cer")]:
            key = f"{prefix}_{unit}"
            selected = [r for r in rows if key in r["counts"]]
            if selected and sum(r["counts"][key]["reference_units"] for r in selected) > 0:
                result[f"{prefix}_corpus_{suffix}"] = clustered_interval(
                    [r["counts"][key]["errors"] for r in selected],
                    [r["semantic_group"] for r in selected],
                    samples,
                    seed,
                    [r["counts"][key]["reference_units"] for r in selected],
                )
                result[f"{prefix}_corpus_{suffix}"].update(
                    coverage=len(selected) / len(rows), missing=len(rows) - len(selected)
                )
    return result


def slices(rows, samples=1000, seed=42):
    result = {}
    for field in ["task", "split", "tts_profile", "language"]:
        partitions = defaultdict(list)
        for row in rows:
            partitions[row[field]].append(row)
        result[field] = {name: summarize(part, samples, seed) for name, part in partitions.items()}
    return result


def frechet_distance(reference, generated):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(generated, dtype=np.float64)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or min(len(a), len(b)) < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "Frechet distance needs at least two finite embeddings per set with the same dimension"
        )
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    ca, cb = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    values, vectors = np.linalg.eigh(ca)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    middle = root @ cb @ root
    trace_root = np.sqrt(np.maximum(np.linalg.eigvalsh((middle + middle.T) / 2), 0)).sum()
    value = (mean_a - mean_b) @ (mean_a - mean_b) + np.trace(ca) + np.trace(cb) - 2 * trace_root
    return float(max(0, value))


def compare_reports(before, after, samples=1000, seed=42, allow_partial=False):
    for key in ["metric_version", "dataset_hash", "protocol_hash", "split"]:
        if before[key] != after[key]:
            raise ValueError(f"Evaluation protocols differ: {key}")
    a, b = {r["example_id"]: r for r in before["rows"]}, {r["example_id"]: r for r in after["rows"]}
    if set(a) != set(b) or len(a) != len(before["rows"]) or len(b) != len(after["rows"]):
        raise ValueError("Comparison requires identical unique example IDs")
    output = {}
    for key in sorted({k for row in before["rows"] + after["rows"] for k in row["metrics"]}):
        missing_a = {i for i in a if a[i]["metrics"].get(key) is None}
        missing_b = {i for i in b if b[i]["metrics"].get(key) is None}
        if missing_a != missing_b and not allow_partial:
            raise ValueError(
                f"Coverage differs for {key}; inspect failures or explicitly allow partial pairs"
            )
        ids = sorted(set(a) - missing_a - missing_b)
        if not ids:
            continue
        if any(a[i]["semantic_group"] != b[i]["semantic_group"] for i in ids):
            raise ValueError("Semantic groups differ")
        report = clustered_interval(
            [b[i]["metrics"][key] - a[i]["metrics"][key] for i in ids],
            [a[i]["semantic_group"] for i in ids],
            samples,
            seed,
        )
        report.update(omitted_pairs=len(a) - len(ids), delta="after_minus_before")
        output[key] = report
    # Corpus WER/CER deltas resample group-level numerator/denominator totals,
    # rather than averaging utterance error rates.
    for prefix in ["text_answer", "spoken_answer", "tts", "input_asr"]:
        for unit, suffix in [("words", "wer"), ("characters", "cer")]:
            key = f"{prefix}_{unit}"
            ids_a = {i for i in a if key in a[i]["counts"]}
            ids_b = {i for i in b if key in b[i]["counts"]}
            if ids_a != ids_b and not allow_partial:
                raise ValueError(f"Coverage differs for {key}")
            ids = sorted(ids_a & ids_b)
            if not ids:
                continue
            refs_a = [a[i]["counts"][key]["reference_units"] for i in ids]
            refs_b = [b[i]["counts"][key]["reference_units"] for i in ids]
            if refs_a != refs_b and prefix != "tts":
                raise ValueError(f"Reference text differs for {key}")
            if sum(refs_a) and sum(refs_b):
                out = paired_ratios(
                    [a[i]["counts"][key]["errors"] for i in ids],
                    refs_a,
                    [b[i]["counts"][key]["errors"] for i in ids],
                    refs_b,
                    [a[i]["semantic_group"] for i in ids],
                    samples,
                    seed,
                )
                out.update(omitted_pairs=len(a) - len(ids), delta="after_minus_before")
                output[f"{prefix}_corpus_{suffix}"] = out
    return {
        "metric_version": before["metric_version"],
        "n_pairs": len(a),
        "bootstrap_unit": "semantic_group",
        "metrics": output,
        "note": "Corpus BLEU/chrF/TER and FAD are reported separately; these intervals are for per-item metrics and corpus WER/CER.",
    }


def paired_ratios(
    numerators_a, denominators_a, numerators_b, denominators_b, groups, samples=1000, seed=42
):
    """Bootstrap two corpus rates together, allowing different TTS text lengths."""
    arrays = np.asarray([numerators_a, denominators_a, numerators_b, denominators_b], dtype=float)
    if (
        arrays.shape != (4, len(groups))
        or not len(groups)
        or not np.isfinite(arrays).all()
        or (arrays < 0).any()
    ):
        raise ValueError("Invalid paired rate counts")
    positions = {g: i for i, g in enumerate(sorted(set(groups)))}
    totals = np.array(
        [
            np.bincount([positions[g] for g in groups], weights=x, minlength=len(positions))
            for x in arrays
        ]
    )
    if totals[1].sum() <= 0 or totals[3].sum() <= 0:
        raise ValueError("Both rates need positive reference counts")

    def delta(x):
        return float(x[2].sum() / x[3].sum() - x[0].sum() / x[1].sum())

    result = {"value": delta(totals), "n": len(groups), "n_groups": len(positions), "ci95": None}
    if len(positions) >= 2:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(samples):
            x = totals[:, rng.integers(len(positions), size=len(positions))]
            if x[1].sum() and x[3].sum():
                estimates.append(delta(x))
        if estimates:
            result["ci95"] = np.quantile(estimates, [0.025, 0.975]).tolist()
    return result
