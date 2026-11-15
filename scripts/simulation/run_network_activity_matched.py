#!/usr/bin/env python3
"""Run the matched-active-cluster experiment and analysis."""
import argparse
from pathlib import Path
from local_market.activity_matched import run, analyze

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, default=Path('configs/network_activity_matched.yaml'))
parser.add_argument('--output', type=Path, default=Path('results/network_activity_matched'))
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--analyze-only', action='store_true')
args = parser.parse_args()
if not args.analyze_only:
    run(args.config, args.output, args.smoke)
analyze(args.output, args.smoke)
