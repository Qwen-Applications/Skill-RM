from __future__ import annotations

from experiments.jetts_seqko.metrics import compute_metrics, render_metrics_md


def test_jetts_metrics_aggregate_seqko_scores() -> None:
    metrics = compute_metrics(
        {
            "s1": {
                "references": {"pass@1": {"score": 1}, "random@10": {"score": 0.5}},
                "seqko": {"1": {"score": 1}, "2": {"score": 0}},
                "counts": {"pairwise_matches": 2, "invalid_decisions": 0},
            },
            "s2": {
                "references": {"pass@1": {"score": 0}, "random@10": {"score": 0.5}},
                "seqko": {"1": {"score": 0}, "2": {"score": 1}},
                "counts": {"pairwise_matches": 2, "invalid_decisions": 1},
            },
        },
        checkpoints=[1, 2],
    )
    assert metrics["n"] == 2
    assert metrics["scores"]["pass@1"]["mean"] == 0.5
    assert metrics["scores"]["seqko@2"]["mean"] == 0.5
    assert metrics["rates"]["invalid_decision_rate"] == 0.25
    table = render_metrics_md({"baseline/gsm8k": metrics})
    assert "SeqKO@2" in table

