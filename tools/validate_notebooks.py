"""Validate and execute each lesson in a fresh kernel, reporting all failures.

Run from the project environment: python -m tools.validate_notebooks --in-place
By default executed copies go into outputs/executed and sources stay untouched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def check(path, in_place=False):
    started = time.perf_counter()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=180, kernel_name="opencv-practical", resources={"metadata": {"path": str(ROOT / "notebooks")}})
    result = {"notebook": path.name, "passed": False}
    try:
        client.execute()
        result["passed"] = True
        result["code_cells"] = sum(cell.cell_type == "code" for cell in notebook.cells)
    except Exception as exc:
        result["error"] = str(exc)
    destination = path if in_place and result["passed"] else ROOT / "outputs" / "executed" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-place", action="store_true", help="Save successful outputs into lesson notebooks.")
    parser.add_argument("--pattern", nargs="+", default=["*.ipynb"], help="One or more filename globs, e.g. '14_*.ipynb'.")
    parser.add_argument("--report", default="outputs/validation_report.json", help="Report path relative to the project root.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    # Keep notebook tests reproducible and avoid oversubscribing laptop CPUs.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    paths = sorted({path for pattern in args.pattern for path in (ROOT / "notebooks").glob(pattern)})
    if not paths:
        parser.error("No matching notebooks found.")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(check, path, args.in_place): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"notebook": path.name, "passed": False, "error": repr(exc)}
            results.append(result)
            print(f"{'PASS' if result['passed'] else 'FAIL'} {path.name} ({result.get('seconds', '?')}s)", flush=True)
            if not result["passed"]:
                print(result["error"], flush=True)
    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(sorted(results, key=lambda x: x["notebook"]), indent=2), encoding="utf-8")
    print(f"Passed {sum(r['passed'] for r in results)}/{len(results)}. Report: {report}")
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
