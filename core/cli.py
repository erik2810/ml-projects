"""Command-line interface for declarative experiments.

Usage::

    python -m core.cli train experiments/configs/karate_gcn.yaml
    python -m core.cli list
    python -m core.cli list --kind model

The ``train`` subcommand takes a YAML or JSON experiment config, instantiates
the model and dataset by their registered names, and runs :class:`core.Trainer`
with the training hyperparameters from the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Importing ``algorithms`` triggers all @register side-effects.
import algorithms  # noqa: F401
from core import Trainer, dump_config, list_registered, load_config
from core.config.schema import ExperimentConfig
from core.registry import get as registry_get
from core.training.callbacks import Checkpoint, EarlyStopping, ProgressLogger
from core.training.trainer import seed_everything


def _build_callbacks(cfg: ExperimentConfig) -> list:
    callbacks = [ProgressLogger(every=max(1, cfg.training.log_every))]

    es_cfg = cfg.training.early_stopping
    if es_cfg:
        callbacks.append(EarlyStopping(**es_cfg))

    if cfg.training.checkpoint_dir:
        callbacks.append(
            Checkpoint(
                dirpath=Path(cfg.training.checkpoint_dir),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            )
        )
    return callbacks


def _instantiate(cfg: ExperimentConfig):
    model_cls = registry_get(cfg.model.name, kind="model")
    dataset_cls = registry_get(cfg.dataset.name, kind="dataset")
    model = model_cls(**cfg.model.params)
    dataset = dataset_cls(**cfg.dataset.params)
    return model, dataset


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.training.seed is not None:
        seed_everything(cfg.training.seed)

    model, dataset = _instantiate(cfg)
    callbacks = _build_callbacks(cfg)

    trainer = Trainer(
        model=model,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        device=cfg.training.device,
        grad_clip=cfg.training.grad_clip,
        seed=cfg.training.seed,
    )
    result = trainer.fit(dataset)

    print(f"\nFinished: epochs_run={result.epochs_run} stopped_early={result.stopped_early}")
    for key, value in result.best_metrics.items():
        print(f"  best {key}: {value:.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "config": cfg.model_dump(),
            "epochs_run": result.epochs_run,
            "stopped_early": result.stopped_early,
            "best_metrics": result.best_metrics,
            "history": result.history,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {out}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    registered = list_registered(args.kind)
    for kind, names in registered.items():
        print(f"{kind}:")
        for name in names:
            print(f"  {name}")
    return 0


def cmd_scaffold(args: argparse.Namespace) -> int:
    """Emit a template YAML config to stdout or ``--out``."""
    cfg = ExperimentConfig(
        name=args.name,
        description="auto-generated template",
        model={"name": args.model, "params": {}},
        dataset={"name": args.dataset, "params": {}},
    )
    if args.out:
        dump_config(cfg, args.out)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(cfg.model_dump(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="core.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run an experiment from a config file.")
    p_train.add_argument("config", type=Path, help="Path to YAML or JSON config.")
    p_train.add_argument("--output", type=Path, help="Write history + metrics to JSON.")
    p_train.set_defaults(func=cmd_train)

    p_list = sub.add_parser("list", help="List registered components.")
    p_list.add_argument("--kind", help="Restrict to one kind (model/dataset/...).")
    p_list.set_defaults(func=cmd_list)

    p_scaffold = sub.add_parser("scaffold", help="Emit a template config.")
    p_scaffold.add_argument("--name", default="new_experiment")
    p_scaffold.add_argument("--model", required=True, help="Registered model name.")
    p_scaffold.add_argument("--dataset", required=True, help="Registered dataset name.")
    p_scaffold.add_argument("--out", type=Path, help="Write to this path.")
    p_scaffold.set_defaults(func=cmd_scaffold)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
