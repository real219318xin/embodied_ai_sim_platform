"""阶段八：批量实验（多随机种子）与结果导出。

用法:
    python scripts/run_experiment.py                     # 默认 3 个种子 × 2 次
    python scripts/run_experiment.py --seeds 1 2 3 --trials 3
    python scripts/run_experiment.py --seeds 7 --trials 1 --jitter 0.02

产出:
    results/experiments/<run_id>/trial_XXX.json
    results/experiment.jsonl
    results/experiment.csv
    results/summary.csv
    results/plots/*.png
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.utils import dump_json                      # noqa: E402
from evaluation.metrics import summarize               # noqa: E402
from evaluation.runner import (DEFAULT_INSTRUCTION, make_plots,  # noqa: E402
                               make_run_id, run_trial)

RES = os.path.join(ROOT, "results")


def write_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--trials", type=int, default=2, help="每个种子的 trial 数")
    ap.add_argument("--jitter", type=float, default=0.015,
                    help="物体/目标初始位置随机范围 (m)")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--shots", action="store_true", help="额外保存搬运阶段截图")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    run_id = make_run_id(args.tag)
    detail_dir = os.path.join(RES, "experiments", run_id)
    os.makedirs(detail_dir, exist_ok=True)
    shots_dir = os.path.join(ROOT, "screenshots")
    os.makedirs(shots_dir, exist_ok=True)

    print("=" * 70)
    print("run_id   :", run_id)
    print("指令     :", args.instruction)
    print("种子     :", args.seeds, "| 每种子 trial:", args.trials,
          "| 位置抖动: ±%.1f mm" % (args.jitter * 1000))
    print("=" * 70)

    records, rows = [], []
    grid = [(s, t) for s in args.seeds for t in range(1, args.trials + 1)]
    for i, (seed, trial) in enumerate(grid, start=1):
        t0 = time.time()
        rec, detail = run_trial(run_id, trial, seed, args.instruction,
                                jitter=args.jitter, shots_dir=shots_dir,
                                save_images=args.shots)
        records.append(rec)
        row = rec.as_row()
        rows.append(row)
        trial_path = os.path.join(detail_dir, f"trial_{i:03d}_seed{seed}.json")
        detail["wall_time_s"] = round(time.time() - t0, 2)
        dump_json(detail, trial_path)
        print(f"[{i:02d}/{len(grid)}] seed={seed} trial={trial} "
              f"success={rec.success} dur={rec.duration_s:5.2f}s "
              f"path={rec.tcp_path_length_m:5.3f}m "
              f"Fmax={rec.peak_grip_force_n:5.2f}N "
              f"coll={rec.collision_events} retry={rec.retries}"
              + (f" xy_err={rec.xy_error_mm:.1f}mm" if rec.xy_error_mm else "")
              + ("" if rec.success else f" | {rec.failed_node}: {rec.reason}"))

    fields = list(rows[0].keys())
    write_jsonl(os.path.join(RES, "experiment.jsonl"), rows)
    write_csv(os.path.join(RES, "experiment.csv"), rows, fields)

    summary_rows = []
    overall = summarize(records)
    summary_rows.append({"scope": "all", "seed": "", **overall})
    for seed in args.seeds:
        s = summarize([r for r in records if r.seed == seed])
        summary_rows.append({"scope": "seed", "seed": seed, **s})
    sfields = list(summary_rows[0].keys())
    write_csv(os.path.join(RES, "summary.csv"), summary_rows, sfields)

    plots = make_plots(records, os.path.join(RES, "plots"))
    dump_json({"run_id": run_id, "instruction": args.instruction,
               "seeds": args.seeds, "trials_per_seed": args.trials,
               "jitter_m": args.jitter, "summary": summary_rows,
               "plots": [os.path.relpath(p, ROOT) for p in plots]},
              os.path.join(RES, "experiment_summary.json"))

    print("-" * 70)
    print("成功 %d/%d  成功率 %.1f%%  平均时长 %.2f s  平均路程 %.3f m"
          % (overall["n_success"], overall["n_trials"],
             overall["success_rate"] * 100, overall["mean_duration_s"],
             overall["mean_tcp_path_length_m"]))
    print("夹持力峰值 %.2f N  碰撞事件 %d  重试 %.2f 次/trial"
          % (overall["max_peak_grip_force_n"], overall["total_collision_events"],
             overall["mean_retries"]))
    if overall["mean_xy_error_mm"] is not None:
        print("放置误差 平均 %.2f mm / 最大 %.2f mm"
              % (overall["mean_xy_error_mm"], overall["max_xy_error_mm"]))
    print("结果目录:", detail_dir)
    print("汇总文件:", os.path.join(RES, "summary.csv"))
    return 0 if overall["success_rate"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
