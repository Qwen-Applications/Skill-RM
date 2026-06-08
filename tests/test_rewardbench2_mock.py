from __future__ import annotations

from skillrm.benchmarks.rewardbench2.metrics import official_metrics_from_rows
from skillrm.benchmarks.rewardbench2.prompts import official_format_ranking_record
from skillrm.runners import rewardbench2


def test_rewardbench2_official_mock_judge(monkeypatch) -> None:
    record = {
        "id": "rb2_mock_1",
        "subset": "Math",
        "prompt": "What is 2 + 2?",
        "chosen": ["4"],
        "rejected": ["3", "5", "22"],
    }
    chosen_label = official_format_ranking_record(record, seed=0)["chosen_label"]

    def fake_call(base_url, messages, config, *, tools=None, tool_choice=None):
        assert base_url == "http://mock/v1"
        assert tools is None
        assert tool_choice is None
        assert config["enable_thinking"] is False
        assert config["send_thinking_field"] is True
        assert messages[0]["role"] == "system"
        return {
            "content": f"The best answer is [[{chosen_label}]].",
            "latency_sec": 0.01,
            "thinking_field_sent": True,
            "reasoning_len": 0,
            "finish_reason": "stop",
            "error": None,
        }

    monkeypatch.setattr(rewardbench2, "call_with_retries", fake_call)
    row = rewardbench2.judge_official_ranking(
        record,
        "http://mock/v1",
        {
            "model": "mock",
            "seed": 0,
            "enable_thinking": False,
            "send_thinking_field": True,
        },
    )

    assert row["sample_id"] == "rb2_mock_1"
    assert row["predicted_label"] == chosen_label
    assert row["official_score"] == 1.0
    assert row["valid"] is True

    metrics = official_metrics_from_rows([record], [row])
    assert metrics["completed"] == 1
    assert metrics["official_leaderboard_domains"]["Math"] == 1.0

