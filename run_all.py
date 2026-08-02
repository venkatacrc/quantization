#!/usr/bin/env python3
"""Run every lab in this curriculum, in order, each as its own subprocess.

This is meant to be run on the GPU node (b200-72), after you've synced this
repo there and installed `requirements-extra.txt` -- see the top-level
README for setup. Running the whole curriculum end to end is a genuinely
long job (expect a few hours in total, dominated by lab 10's capstone), so
this script is built to survive individual lab failures, log everything to
disk, and let you scope a run down to exactly the labs you want.

Examples:

    # Run absolutely everything, in order.
    python run_all.py

    # Just see what would run, without running anything.
    python run_all.py --dry-run

    # Run only labs 00-04 (the from-scratch PTQ methods).
    python run_all.py --only 00,01,02,03,04

    # Run everything except the slow capstone and the optional
    # "production library" comparison scripts (lab02b, lab03b, lab05b).
    python run_all.py --skip 10 --skip-production-comparisons

    # Resume a run that died partway through, starting at lab 06.
    python run_all.py --from 06

Each lab's stdout/stderr is streamed live to your terminal *and* saved to
`run_all_logs/<lab>__<script>.log`. A `run_all_logs/summary.json` and a
plain-text table are written at the end (and updated after every lab, so a
run that gets interrupted still leaves a useful partial summary).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "run_all_logs"


@dataclass
class LabScript:
    lab_dir: str       # e.g. "02_gptq"
    script: str        # e.g. "lab02_gptq_from_scratch.py"
    production_comparison: bool = False  # the "b" scripts that need extra optional libs

    @property
    def lab_number(self) -> str:
        return self.lab_dir[:2]

    @property
    def path(self) -> Path:
        return REPO_ROOT / "labs" / self.lab_dir / self.script

    @property
    def log_name(self) -> str:
        return f"{self.lab_dir}__{self.script.removesuffix('.py')}.log"


ALL_SCRIPTS: list[LabScript] = [
    LabScript("00_fundamentals", "lab00_numerics.py"),
    LabScript("01_rtn_weight_only", "lab01_rtn.py"),
    LabScript("02_gptq", "lab02_gptq_from_scratch.py"),
    LabScript("02_gptq", "lab02b_compare_production_gptq.py", production_comparison=True),
    LabScript("03_awq", "lab03_awq_from_scratch.py"),
    LabScript("03_awq", "lab03b_compare_production_awq.py", production_comparison=True),
    LabScript("04_smoothquant", "lab04_smoothquant.py"),
    LabScript("05_fp8", "lab05_fp8_inference.py"),
    LabScript("05_fp8", "lab05b_compare_torchao.py", production_comparison=True),
    LabScript("06_fp4_nvfp4", "lab06_fp4_blackwell.py"),
    LabScript("07_qat", "lab07_qat_lsq.py"),
    LabScript("08_qlora", "lab08_qlora_finetune.py"),
    LabScript("09_kv_cache_quant", "lab09_kv_cache_quant.py"),
    LabScript("10_capstone_serving_bench", "lab10_multi_precision_serving_bench.py"),
]


@dataclass
class RunResult:
    script: LabScript
    status: str  # "PASS", "FAIL", "SKIPPED"
    elapsed_s: float = 0.0
    returncode: int | None = None
    log_path: str = ""
    detail: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=str, default=None, help="comma-separated lab numbers to run, e.g. '00,02,05'")
    p.add_argument("--skip", type=str, default=None, help="comma-separated lab numbers to skip, e.g. '08,10'")
    p.add_argument("--from", dest="from_lab", type=str, default=None, help="start at this lab number (inclusive)")
    p.add_argument("--to", dest="to_lab", type=str, default=None, help="stop after this lab number (inclusive)")
    p.add_argument(
        "--skip-production-comparisons",
        action="store_true",
        help="skip the optional lab02b/lab03b/lab05b scripts that compare against gptqmodel/autoawq/torchao",
    )
    p.add_argument("--stop-on-failure", action="store_true", help="stop the whole run on the first failing lab (default: keep going)")
    p.add_argument("--dry-run", action="store_true", help="print what would run, without running anything")
    p.add_argument("--python", type=str, default=sys.executable, help="python executable to use (default: this one)")
    return p.parse_args()


def select_scripts(args: argparse.Namespace) -> list[LabScript]:
    scripts = list(ALL_SCRIPTS)

    if args.skip_production_comparisons:
        scripts = [s for s in scripts if not s.production_comparison]

    if args.only:
        wanted = {n.strip().zfill(2) for n in args.only.split(",")}
        scripts = [s for s in scripts if s.lab_number in wanted]
    if args.skip:
        excluded = {n.strip().zfill(2) for n in args.skip.split(",")}
        scripts = [s for s in scripts if s.lab_number not in excluded]
    if args.from_lab:
        scripts = [s for s in scripts if s.lab_number >= args.from_lab.strip().zfill(2)]
    if args.to_lab:
        scripts = [s for s in scripts if s.lab_number <= args.to_lab.strip().zfill(2)]

    return scripts


def run_env_check(python: str) -> None:
    print("=" * 78)
    print("  Environment check (informational -- a missing optional package just")
    print("  means the labs that need it will report that and skip cleanly)")
    print("=" * 78)
    subprocess.run([python, "-m", "common.env_check"], cwd=REPO_ROOT)
    print()


def run_and_tee(cmd: list[str], log_path: Path) -> int:
    """Run `cmd`, streaming output to both stdout and `log_path` live."""
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
        return process.returncode


def write_summary(results: list[RunResult]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_json = [
        {
            "lab": r.script.lab_dir,
            "script": r.script.script,
            "status": r.status,
            "elapsed_s": round(r.elapsed_s, 1),
            "returncode": r.returncode,
            "log": r.log_path,
        }
        for r in results
    ]
    (LOG_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2))

    lines = [f"{'STATUS':<8} {'ELAPSED':>9}  {'LAB':<28} SCRIPT"]
    for r in results:
        lines.append(f"{r.status:<8} {r.elapsed_s:>8.1f}s  {r.script.lab_dir:<28} {r.script.script}")
    (LOG_DIR / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    scripts = select_scripts(args)

    if not scripts:
        print("No scripts matched your --only/--skip/--from/--to filters. Nothing to do.")
        return

    print("Planned run order:")
    for s in scripts:
        tag = " (production comparison, optional deps)" if s.production_comparison else ""
        print(f"  labs/{s.lab_dir}/{s.script}{tag}")
    print()

    if args.dry_run:
        print("(--dry-run: not actually running anything)")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_env_check(args.python)

    results: list[RunResult] = []
    overall_start = time.time()

    for i, s in enumerate(scripts, start=1):
        print("=" * 78)
        print(f"  [{i}/{len(scripts)}] labs/{s.lab_dir}/{s.script}")
        print("=" * 78)

        if not s.path.exists():
            print(f"  script not found at {s.path}, skipping.")
            results.append(RunResult(s, "SKIPPED", detail={"reason": "script not found"}))
            write_summary(results)
            continue

        log_path = LOG_DIR / s.log_name
        t0 = time.time()
        returncode = run_and_tee([args.python, str(s.path)], log_path)
        elapsed = time.time() - t0

        status = "PASS" if returncode == 0 else "FAIL"
        results.append(RunResult(s, status, elapsed_s=elapsed, returncode=returncode, log_path=str(log_path)))
        write_summary(results)

        print(f"\n  -> {status} in {elapsed:.1f}s (log: {log_path})\n")
        if status == "FAIL" and args.stop_on_failure:
            print("Stopping (--stop-on-failure was set).")
            break

    total_elapsed = time.time() - overall_start
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIPPED")

    print("=" * 78)
    print(f"  Done in {total_elapsed / 60:.1f} min: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"  Full summary: {LOG_DIR / 'summary.txt'}  (and summary.json for scripting)")
    print("=" * 78)
    for r in results:
        print(f"  {r.status:<8} {r.elapsed_s:>8.1f}s  labs/{r.script.lab_dir}/{r.script.script}")


if __name__ == "__main__":
    main()
