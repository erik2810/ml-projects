"""Smoke tests for the experiment CLI.

Runs the CLI end-to-end as library calls (no subprocess) to keep tests fast
and properly traceable. Uses the tiny ``spring_mesh`` + ``physics_gnn_node``
combo so a single epoch completes in well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.cli import main as cli_main


def _minimal_config(path: Path) -> Path:
    cfg = {
        "name": "cli_smoke",
        "description": "pytest smoke",
        "model": {
            "name": "gcn_node",
            "params": {
                "in_features": 18,
                "num_classes": 2,
                "hidden": 8,
                "n_layers": 2,
                "dropout": 0.0,
                "lr": 0.01,
            },
        },
        "dataset": {"name": "karate_club", "params": {"per_class": 4, "seed": 0}},
        "training": {
            "max_epochs": 3,
            "seed": 0,
            "device": "cpu",
            "log_every": 5,
        },
    }
    path.write_text(json.dumps(cfg))
    return path


def test_cli_list_runs(capsys):
    ret = cli_main(["list"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "gcn_node" in out
    assert "karate_club" in out


def test_cli_list_filter(capsys):
    ret = cli_main(["list", "--kind", "dataset"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "karate_club" in out
    # Model names should not appear when filtering to datasets.
    assert "gcn_node" not in out


def test_cli_train_end_to_end(tmp_path: Path, capsys):
    cfg_path = _minimal_config(tmp_path / "cfg.json")
    out_path = tmp_path / "result.json"
    ret = cli_main(["train", str(cfg_path), "--output", str(out_path)])
    assert ret == 0
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    assert payload["epochs_run"] == 3
    assert "train_loss" in payload["history"]
    assert "best_metrics" in payload


def test_cli_scaffold_writes_file(tmp_path: Path):
    out = tmp_path / "scaffold.yaml"
    pytest.importorskip("yaml")
    ret = cli_main([
        "scaffold",
        "--model", "gcn_node",
        "--dataset", "karate_club",
        "--out", str(out),
    ])
    assert ret == 0
    assert out.exists()
    contents = out.read_text()
    assert "gcn_node" in contents
    assert "karate_club" in contents
