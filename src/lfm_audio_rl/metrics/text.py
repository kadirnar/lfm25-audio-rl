import re
import unicodedata
from collections import Counter

from ..data import normalize as dataset_normalize


def normalize(text: str, mode="basic"):
    if mode == "dataset":
        return dataset_normalize(text)
    text = unicodedata.normalize("NFKC", text)
    if mode == "basic":
        text = "".join(
            " " if unicodedata.category(c).startswith("P") else c for c in text.casefold()
        )
    elif mode != "verbatim":
        raise ValueError("Unknown normalization")
    return " ".join(text.split())


def edit_counts(reference, hypothesis):
    # Keep a deterministic S/D/I breakdown, not just a scalar distance.
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, a in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, b in enumerate(hypothesis, 1):
            if a == b:
                current.append(previous[j - 1])
                continue
            cost, s, d, ins = previous[j - 1]
            sub = (cost + 1, s + 1, d, ins)
            cost, s, d, ins = previous[j]
            delete = (cost + 1, s, d + 1, ins)
            cost, s, d, ins = current[j - 1]
            insert = (cost + 1, s, d, ins + 1)
            current.append(min([sub, delete, insert], key=lambda x: x[0]))
        previous = current
    errors, s, d, ins = previous[-1]
    n = len(reference)
    return {
        "errors": errors,
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "reference_units": n,
        "rate": errors / n if n else (0.0 if not hypothesis else None),
    }


def text_scores(reference: str, hypothesis: str, mode="basic", alternatives=()):
    ref, hyp = normalize(reference, mode), normalize(hypothesis, mode)
    refs = [ref, *(normalize(a, mode) for a in alternatives)]
    em = float(hyp in refs)
    scores = []
    for r in refs:
        a, b = r.split(), hyp.split()
        overlap = sum((Counter(a) & Counter(b)).values())
        scores.append(2 * overlap / (len(a) + len(b)) if a or b else 1.0)
    words = edit_counts(ref.split(), hyp.split())
    chars = edit_counts(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    return {
        "exact_match": em,
        "token_f1": max(scores),
        "wer": words["rate"],
        "cer": chars["rate"],
    }, {"words": words, "characters": chars}


def corpus_text(references: list[str], hypotheses: list[str]):
    from rouge_score.rouge_scorer import RougeScorer
    from sacrebleu.metrics import BLEU, CHRF, TER

    if len(references) != len(hypotheses) or not references:
        raise ValueError("Need matching nonempty text collections")
    metrics = {"bleu": BLEU(), "chrf": CHRF(), "ter": TER()}
    result = {}
    for name, metric in metrics.items():
        score = metric.corpus_score(hypotheses, [references])
        result[name] = {"value": score.score, "signature": str(metric.get_signature())}
    rouge = RougeScorer(["rougeL"], use_stemmer=False)
    result["rouge_l_f1"] = {
        "value": sum(
            rouge.score(r, h)["rougeL"].fmeasure
            for r, h in zip(references, hypotheses, strict=True)
        )
        / len(references),
        "signature": "rouge-score/rougeL/no-stemmer/macro",
    }
    return result


def diversity(texts: list[str], mode="basic"):
    tokens = [normalize(t, mode).split() for t in texts]
    result = {}
    for n in [1, 2, 3]:
        grams = [tuple(words[i : i + n]) for words in tokens for i in range(len(words) - n + 1)]
        result[f"distinct_{n}"] = len(set(grams)) / len(grams) if grams else None
        result[f"repeated_{n}_gram_fraction"] = 1 - len(set(grams)) / len(grams) if grams else None
    normalized = [normalize(t, mode) for t in texts]
    result["duplicate_response_fraction"] = (
        1 - len(set(normalized)) / len(normalized) if normalized else None
    )
    return result


def constraint_scores(text, constraints, mode="basic"):
    if not constraints:
        return {}
    text = normalize(text, mode)
    passed = []
    for c in constraints:
        v = normalize(c.value, mode) if isinstance(c.value, str) else c.value
        # Phrase matching avoids counting 'red' inside 'hundred'.
        contains = (
            bool(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", text))
            if isinstance(v, str)
            else False
        )
        if c.kind == "contains":
            valid = contains
        elif c.kind == "not_contains":
            valid = not contains
        elif c.kind == "starts_with":
            valid = text.startswith(v)
        elif c.kind == "ends_with":
            valid = text.endswith(v)
        elif c.kind == "exact":
            valid = text == v
        elif c.kind == "min_words":
            valid = len(text.split()) >= v
        else:
            valid = len(text.split()) <= v
        passed.append(valid)
    return {
        "instruction_pass_rate": sum(passed) / len(passed),
        "instruction_all_pass": float(all(passed)),
    }
