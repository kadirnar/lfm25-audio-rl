from collections import defaultdict
from typing import Literal

from pydantic import Field

from ..config import StrictModel
from .statistics import clustered_interval


class Rating(StrictModel):
    example_id: str
    rater_id: str
    dimension: Literal[
        "naturalness", "intelligibility", "speaker_similarity", "helpfulness", "prosody"
    ]
    score: int = Field(ge=1, le=5)


class Preference(StrictModel):
    example_id: str
    rater_id: str
    winner: Literal["A", "B", "tie"]


def summarize_ratings(records, kind="mos", groups=None, samples=1000, seed=42):
    cls = Rating if kind == "mos" else Preference if kind == "preference" else None
    if cls is None:
        raise ValueError("Unknown listening-test format")
    ratings = [cls.model_validate(r) for r in records]
    if not ratings:
        raise ValueError("No human ratings")
    seen, by_dimension = set(), defaultdict(lambda: defaultdict(list))
    for r in ratings:
        dimension = r.dimension if kind == "mos" else "B_preference_with_half_ties"
        key = (r.example_id, r.rater_id, dimension)
        if key in seen:
            raise ValueError("Duplicate rating by the same rater")
        seen.add(key)
        if groups is not None and r.example_id not in groups:
            raise ValueError("Rating refers to an unknown example")
        score = r.score if kind == "mos" else {"A": 0.0, "B": 1.0, "tie": 0.5}[r.winner]
        by_dimension[dimension][r.example_id].append(score)
    result = {}
    for dimension, items in by_dimension.items():
        ids = sorted(items)
        result[dimension] = clustered_interval(
            [sum(items[i]) / len(items[i]) for i in ids],
            [groups[i] if groups else i for i in ids],
            samples,
            seed,
        )
        result[dimension]["n_ratings"] = sum(map(len, items.values()))
    return {
        "kind": kind,
        "n_raters": len({r.rater_id for r in ratings}),
        "bootstrap_unit": "semantic_group" if groups else "example_id",
        "dimensions": result,
    }
