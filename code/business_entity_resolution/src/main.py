#!/usr/bin/env python3
"""
CLI entrypoint for the Business Entity Resolution pipeline.

    python3 main.py train  --data-dir dataset --model-dir models
    python3 main.py infer  --data-dir dataset --model-dir models --output-dir output
    python3 main.py generate-sample-data --out-dir dataset

See README.md for full usage, ablation flags, and reproduction steps.
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import PipelineConfig  # noqa: E402


def _apply_ablation_flags(cfg: PipelineConfig, args) -> PipelineConfig:
    if args.no_llr:
        cfg.use_llr_feature = False
    if args.no_triangulation:
        cfg.use_triangulation_feature = False
    if args.no_hard_negative_mining:
        cfg.use_hard_negative_mining = False
    if args.no_calibration:
        cfg.use_calibration = False
    if args.no_existence_model:
        cfg.use_existence_model = False
    if args.no_metric_decoder:
        cfg.use_metric_aware_decoder = False
    return cfg


def build_parser():
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-llr", action="store_true", help="ablation: disable Fellegi-Sunter LLR feature")
    common.add_argument("--no-triangulation", action="store_true", help="ablation: disable cross-source triangulation feature")
    common.add_argument("--no-hard-negative-mining", action="store_true", help="ablation: disable hard-negative mining")
    common.add_argument("--no-calibration", action="store_true", help="ablation: disable probability calibration")
    common.add_argument("--no-existence-model", action="store_true", help="ablation: disable singleton/existence model")
    common.add_argument("--no-metric-decoder", action="store_true", help="ablation: use naive p>0.5 threshold instead of the metric-aware F0.5 decoder")
    common.add_argument("--log-level", default="INFO")

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--data-dir", required=True, help="directory containing train/ (and test/)")
    t.add_argument("--model-dir", required=True)

    i = sub.add_parser("infer", parents=[common])
    i.add_argument("--data-dir", required=True, help="directory containing test/")
    i.add_argument("--model-dir", required=True)
    i.add_argument("--output-dir", required=True)

    g = sub.add_parser("generate-sample-data")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--n-train", type=int, default=400)
    g.add_argument("--n-test", type=int, default=200)
    g.add_argument("--seed", type=int, default=13)

    v = sub.add_parser("validate")
    v.add_argument("--matching", required=True)
    v.add_argument("--candidate", required=True)
    v.add_argument("--test-dir", required=True)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, getattr(args, "log_level", "INFO")),
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.command == "train":
        cfg = _apply_ablation_flags(PipelineConfig(), args)
        from pipeline.train import run_training
        metrics = run_training(args.data_dir, args.model_dir, cfg)
        print(json.dumps(metrics, indent=2))

    elif args.command == "infer":
        from pipeline.infer import run_inference
        summary = run_inference(args.data_dir, args.model_dir, args.output_dir)
        print(json.dumps(summary, indent=2))

    elif args.command == "generate-sample-data":
        from data_synth.generate_sample_data import generate
        generate(args.out_dir, n_train=args.n_train, n_test=args.n_test, seed=args.seed)

    elif args.command == "validate":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        from validate_submission import validate
        ok = validate(args.matching, args.candidate, args.test_dir)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
