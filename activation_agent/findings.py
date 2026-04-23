"""
Structured findings object. This is the bridge between the analytical
layer (funnel.py, cohorts.py) and the narrative layer (diagnose.py).

The LLM sees ONLY the fields in this object — not the raw event data.
That's the whole point of the separation: the LLM interprets, it doesn't
compute.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .cohorts import DivergentSegment
from .funnel import FunnelSummary


@dataclass
class Findings:
    overall: FunnelSummary
    divergent_segments: List[DivergentSegment] = field(default_factory=list)
    attribute_columns_analyzed: List[str] = field(default_factory=list)
    # How many segments existed per attribute column that were too small
    # to analyze — useful context for the LLM (so it doesn't over-claim
    # "no divergent segments in X" when we just didn't have the data).
    small_segment_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict. Used for the LLM prompt and for inspection."""
        return {
            "overall": {
                "total_users": self.overall.total_users,
                "end_to_end_conversion": self.overall.end_to_end_conversion,
                "step_order": self.overall.step_order,
                "steps": [asdict(s) for s in self.overall.steps],
                "biggest_drop_off_step": (
                    asdict(self.overall.biggest_drop_off_step)
                    if self.overall.biggest_drop_off_step
                    else None
                ),
            },
            "divergent_segments": [
                {
                    "segment_column": seg.segment_column,
                    "segment_value": seg.segment_value,
                    "segment_n": seg.segment_n,
                    "segment_end_to_end": seg.segment_end_to_end,
                    "overall_end_to_end": seg.overall_end_to_end,
                    "end_to_end_delta_pp": seg.end_to_end_delta_pp,
                    "direction": seg.divergence_direction,
                    "divergent_steps": [asdict(sd) for sd in seg.divergent_steps],
                }
                for seg in self.divergent_segments
            ],
            "attribute_columns_analyzed": self.attribute_columns_analyzed,
            "small_segment_counts": self.small_segment_counts,
        }
