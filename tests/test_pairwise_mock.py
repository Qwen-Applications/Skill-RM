from __future__ import annotations

from skillrm.benchmarks.pairwise.data import OpenRSPairTask
from skillrm.benchmarks.pairwise.metrics import compute_openrs_metrics
from skillrm.runners import pairwise


def test_judgebench_pairwise_mock_baseline(monkeypatch) -> None:
    task = OpenRSPairTask(
        task_id="jb_1::order1",
        benchmark="judgebench_gpt",
        sample_id="jb_1",
        prompt="Pick the response that answers the math question.",
        responses={"A": "The answer is 4.", "B": "The answer is 5."},
        gold_label="A",
        query_type="math",
        domain="math",
        order=1,
    )

    def fake_call(base_url, messages, config, *, tools=None, tool_choice=None):
        assert base_url == "http://mock/v1"
        assert config["enable_thinking"] is False
        assert config["send_thinking_field"] is True
        assert messages
        return {
            "content": "Output (a)",
            "latency_sec": 0.01,
            "thinking_field_sent": True,
            "reasoning_len": 0,
            "finish_reason": "stop",
            "tool_calls": [],
            "error": None,
        }

    monkeypatch.setattr(pairwise, "call_with_retries", fake_call)
    row = pairwise.judge_pairwise_baseline(
        task,
        "http://mock/v1",
        {
            "benchmark": "judgebench_gpt",
            "enable_thinking": False,
            "send_thinking_field": True,
        },
    )

    assert row["predicted_label"] == "A"
    assert row["valid"] is True
    assert row["correct"] is True

    metrics = compute_openrs_metrics(
        [task],
        [row],
        {"benchmark": "judgebench_gpt", "evaluation_mode": "pairwise_baseline"},
    )
    assert metrics["overall"]["acc_rate"] == 1.0
    assert metrics["task_level"]["overall"]["invalid_rate"] == 0.0

