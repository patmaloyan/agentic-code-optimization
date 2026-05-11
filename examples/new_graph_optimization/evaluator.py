"""
Evaluator for the shortest path optimization example.

This evaluator runs the shortest path program and scores it based on:
1. Correctness
2. Performance
3. Scalability across input sizes
4. Behavior across graph regimes: sparse, medium, dense
"""

import sys
import subprocess
import re


def evaluate(program_path: str) -> dict:
    """
    Evaluate the shortest path program.

    Args:
        program_path: Path to the program to evaluate

    Returns:
        Dictionary with:
            - metrics: dict with combined_score and other metrics
            - artifacts: dict with timing details and output
    """
    try:
        result = subprocess.run(
            [sys.executable, program_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {
                "metrics": {"combined_score": 0.0},
                "artifacts": {"error": result.stderr}
            }

        output = result.stdout

        if "ERROR" in output:
            return {
                "metrics": {"combined_score": 0.0},
                "artifacts": {"error": "Shortest path correctness error", "output": output}
            }

        timings = []
        regime_times = {
            "sparse": [],
            "medium": [],
            "dense": []
        }

        # Expected timing line:
        # Regime sparse Size   50 Edges    150: 0.123 ms
        timing_pattern = re.compile(
            r"Regime\s+(\w+)\s+Size\s+(\d+)\s+Edges\s+(\d+):\s+(\d+\.\d+)\s*ms"
        )

        for line in output.splitlines():
            match = timing_pattern.search(line)

            if match:
                regime = match.group(1)
                size = int(match.group(2))
                edges = int(match.group(3))
                time_ms = float(match.group(4))

                record = {
                    "regime": regime,
                    "size": size,
                    "edges": edges,
                    "time_ms": time_ms
                }

                timings.append(record)

                if regime in regime_times:
                    regime_times[regime].append(record)

        total_time = None

        for line in output.splitlines():
            if "Total time:" in line:
                match = re.search(r"(\d+\.\d+)\s*ms", line)

                if match:
                    total_time = float(match.group(1))
                    break

        if total_time is None or not timings:
            return {
                "metrics": {"combined_score": 0.0},
                "artifacts": {
                    "error": "Could not parse timing information",
                    "output": output
                }
            }

        # Baseline estimate:
        # The initial adjacency-matrix Dijkstra is expected to be decent on small
        # inputs but scale poorly on sparse/medium graphs. This baseline should
        # be adjusted after observing the first run of the initial program.
        baseline_time = 1000.0

        performance_score = baseline_time / (baseline_time + total_time)

        # Scalability score:
        # For each regime, compare largest-size timing to smallest-size timing.
        # The score adapts to the actual size ratio used by the benchmark.
        # O(V^2) behavior would suggest about (size_ratio)^2 growth.
        # Better sparse-graph algorithms should grow much more slowly.
        scalability_scores = []

        for regime, times in regime_times.items():
            if len(times) < 2:
                continue

            first = times[0]
            last = times[-1]
            size_ratio = last["size"] / first["size"] if first["size"] > 0 else float("inf")

            if first["time_ms"] <= 0:
                ratio = float("inf")
            else:
                ratio = last["time_ms"] / first["time_ms"]

            ideal_ratio = size_ratio
            worst_ratio = size_ratio ** 2

            if ratio <= ideal_ratio:
                score = 1.0
            elif ratio >= worst_ratio:
                score = 0.0
            else:
                score = 1.0 - (ratio - ideal_ratio) / (worst_ratio - ideal_ratio)

            scalability_scores.append(score)

        if scalability_scores:
            scalability_score = sum(scalability_scores) / len(scalability_scores)
        else:
            scalability_score = 0.5

        # Sparse-regime score:
        # This specifically rewards improvements where adjacency lists and heaps
        # help sparse graphs more than the dense case.
        sparse_score = 0.5

        if len(regime_times["sparse"]) >= 2:
            sparse_first = regime_times["sparse"][0]
            sparse_last = regime_times["sparse"][-1]

            if sparse_first["time_ms"] > 0:
                sparse_ratio = sparse_last["time_ms"] / sparse_first["time_ms"]
                size_ratio = sparse_last["size"] / sparse_first["size"]
                worst_ratio = size_ratio ** 2

                if sparse_ratio <= size_ratio:
                    sparse_score = 1.0
                elif sparse_ratio >= worst_ratio:
                    sparse_score = 0.0
                else:
                    sparse_score = 1.0 - (sparse_ratio - size_ratio) / (worst_ratio - size_ratio)

        # Regime balance score:
        # Prevents an optimizer from only doing well on one regime while becoming
        # terrible on another.
        regime_totals = {}

        for regime, times in regime_times.items():
            if times:
                regime_totals[regime] = sum(times)

        if len(regime_totals) == 3:
            slowest = max(regime_totals.values())
            fastest = min(regime_totals.values())

            if fastest <= 0:
                regime_balance_score = 0.0
            else:
                imbalance = slowest / fastest

                # Some imbalance is expected because dense graphs are harder.
                # Very high imbalance means the implementation is regime-fragile.
                if imbalance <= 4.0:
                    regime_balance_score = 1.0
                elif imbalance >= 20.0:
                    regime_balance_score = 0.0
                else:
                    regime_balance_score = 1.0 - (imbalance - 4.0) / 16.0
        else:
            regime_balance_score = 0.5

        combined_score = (
            0.45 * performance_score
            + 0.30 * scalability_score
            + 0.15 * sparse_score
            + 0.10 * regime_balance_score
        )

        return {
            "metrics": {
                "combined_score": float(combined_score),
                "performance_score": float(performance_score),
                "scalability_score": float(scalability_score),
                "sparse_score": float(sparse_score),
                "regime_balance_score": float(regime_balance_score),
                "total_time_ms": float(total_time)
            },
            "artifacts": {
                "timings": timings,
                "regime_times": regime_times,
                "regime_totals": regime_totals,
                "output": output
            }
        }

    except subprocess.TimeoutExpired:
        return {
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": "Program timed out"}
        }

    except Exception as e:
        return {
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": str(e)}
        }