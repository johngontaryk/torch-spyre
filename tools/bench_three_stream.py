# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2-stream vs 3-stream wall-time benchmark.

Simulates a decode projection loop: a small-M matmul (batch=1, large K/N)
compiled and run back-to-back N times without draining between steps.
This is the workload where HostCompute+H2D corrections are largest relative
to compute time, giving 3-stream the best opportunity to show a benefit.

Usage:
    python tools/bench_three_stream.py
    python tools/bench_three_stream.py --steps 200 --m 1 --k 8192 --n 8192
    python tools/bench_three_stream.py --steps 100 --m 4 --k 4096 --n 4096

    # Vary M to sweep compute/correction ratio:
    for m in 1 2 4 8 16 32; do
        python tools/bench_three_stream.py --m $m
    done

Output per topology:
    RESULT topology=two_stream   total_ms=X.X  per_step_ms=X.XXX
    RESULT topology=three_stream total_ms=X.X  per_step_ms=X.XXX
    SPEEDUP three_stream_vs_two_stream=X.XXXx
"""

import argparse
import os
import statistics
import time

import torch
import torch._dynamo
import torch_spyre  # noqa: F401 — registers the spyre backend


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=1, help="Batch/M dim (default: 1)")
    p.add_argument("--k", type=int, default=4096, help="K dim (default: 4096)")
    p.add_argument("--n", type=int, default=4096, help="N dim (default: 4096)")
    p.add_argument(
        "--steps", type=int, default=100, help="Decode steps per trial (default: 100)"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("BENCH_WARMUP", "5")),
        help="Warmup iterations before measurement (default: 5)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=int(os.environ.get("BENCH_REPS", "5")),
        help="Number of full trials to take the median over (default: 5)",
    )
    return p.parse_args()


def _make_fn(k: int, n: int):
    """Decode projection: x @ weight, weight is a fixed device tensor."""
    weight = torch.randn(k, n, dtype=torch.float16).to("spyre")

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x @ weight

    return fn


def _run_trial(compiled_fn, x: torch.Tensor, steps: int) -> float:
    """Run `steps` back-to-back decode steps without draining between them.

    No .cpu() inside the loop — that would block the CPU and kill the overlap.
    We synchronize once at the end so wall time covers the full device pipeline.
    """
    t0 = time.perf_counter()
    for _ in range(steps):
        compiled_fn(x)
    torch.spyre.synchronize()
    return (time.perf_counter() - t0) * 1e3  # ms


def _benchmark(
    topology_name: str,
    env_val: str | None,
    args: argparse.Namespace,
) -> float:
    """Compile and run the workload under the given topology. Returns median total_ms."""
    if env_val is None:
        os.environ.pop("TORCH_SPYRE_THREE_STREAM", None)
    else:
        os.environ["TORCH_SPYRE_THREE_STREAM"] = env_val

    # Fresh compile so each topology gets its own kernel binary.
    torch._dynamo.reset()

    fn = _make_fn(args.k, args.n)
    compiled_fn = torch.compile(fn, backend="inductor")
    x = torch.randn(args.m, args.k, dtype=torch.float16).to("spyre")

    # Warmup: triggers compilation + fills kernel cache.
    for _ in range(args.warmup):
        _run_trial(compiled_fn, x, args.steps)

    # Measurement: reps full trials, report median.
    totals = [_run_trial(compiled_fn, x, args.steps) for _ in range(args.reps)]

    median = statistics.median(totals)
    minimum = min(totals)
    std = statistics.pstdev(totals) if len(totals) > 1 else 0.0
    cv = std / statistics.fmean(totals) * 100.0 if totals else 0.0

    print(
        f"RESULT topology={topology_name:<15s}  "
        f"total_ms={median:.1f}  per_step_ms={median / args.steps:.3f}  "
        f"min_ms={minimum:.1f}  cv={cv:.1f}%"
    )
    return median


def main() -> None:
    args = _parse_args()

    print(
        f"Workload: decode projection  "
        f"M={args.m}  K={args.k}  N={args.n}  "
        f"steps={args.steps}  warmup={args.warmup}  reps={args.reps}"
    )
    print(
        "Note: HostCompute+H2D correction overhead is fixed per step regardless "
        "of M.\n      Small M = corrections dominate = best case for 3-stream."
    )
    print("-" * 70)

    two_stream = _benchmark("two_stream", None, args)
    three_stream = _benchmark("three_stream", "1", args)

    speedup = two_stream / three_stream if three_stream > 0 else float("inf")
    print("-" * 70)
    print(f"SPEEDUP three_stream_vs_two_stream={speedup:.3f}x")
    if speedup > 1.02:
        print("→ 3-stream is FASTER  (>2% improvement)")
    elif speedup < 0.98:
        print("→ 3-stream is SLOWER  (>2% regression)")
    else:
        print("→ no meaningful difference (<2%)")


if __name__ == "__main__":
    main()
