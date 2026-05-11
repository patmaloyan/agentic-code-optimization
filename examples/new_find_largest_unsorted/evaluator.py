"""
Evaluator for the top-k largest numbers optimization example.

This evaluator runs the top-k program and scores it based on:
1. Correctness: must return the k largest numbers in descending order.
2. Performance: lower total runtime is better.
3. Scalability: runtime should grow much better than quadratic.

The target efficient idea is:
    maintain a min-heap of size k, then sort the final k values
    O(n log k + k log k)
"""

import sys
import subprocess
import re


def evaluate(program_path: str) -> dict:
    """
    Evaluate the top-k program.

    Args:
        program_path: Path to the program to evaluate

    Returns:
        Dictionary with:
            - metrics: dict with 'combined_score' and other metrics
            - artifacts: dict with additional information
    """
    try:
        # Run the program and capture output.
        result = subprocess.run(
            [sys.executable, program_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            # Program failed.
            return {
                "metrics": {"combined_score": 0.0},
                "artifacts": {"error": result.stderr},
            }

        output = result.stdout

        # Check for correctness errors.
        if "ERROR" in output:
            return {
                "metrics": {"combined_score": 0.0},
                "artifacts": {"error": "Top-k correctness error", "output": output},
            }

        # Parse timing information.
        times = []
        sizes = []
        ks = []

        for line in output.split("\n"):
            if "Size" in line and "ms" in line:
                size_match = re.search(r"Size\s+(\d+)", line)
                k_match = re.search(r"k\s+(\d+)", line)
                time_match = re.search(r"(\d+\.\d+)\s*ms", line)

                if size_match and k_match and time_match:
                    sizes.append(int(size_match.group(1)))
                    ks.append(int(k_match.group(1)))
                    times.append(float(time_match.group(1)))

        # Extract total time.
        total_time = None
        for line in output.split("\n"):
            if "Total time:" in line:
                match = re.search(r"(\d+\.\d+)\s*ms", line)
                if match:
                    total_time = float(match.group(1))
                    break

        if total_time is None or not times:
            return {
                "metrics": {"combined_score": 0.0},
                "artifacts": {"error": "Could not parse timing information", "output": output},
            }

        # Calculate performance score.
        # Baseline: naive full bubble sort is expected to be slow on these cases.
        # Faster top-k algorithms should score much closer to 1.0.
        baseline_time = 2500.0  # ms

        # Score based on speedup over baseline.
        # Score = baseline / (baseline + actual)
        performance_score = baseline_time / (baseline_time + total_time)

        # Check scalability by comparing growth from the first to last test.
        # Naive bubble sort behaves roughly O(n^2), while good top-k methods
        # should be closer to O(n log k + k log k).
        if len(times) >= 2 and times[0] > 0 and sizes[0] > 0:
            size_ratio = sizes[-1] / sizes[0]
            time_ratio = times[-1] / times[0]

            # Linear-ish growth is ideal for fixed/small k.
            ideal_ratio = size_ratio
            # Quadratic growth is the poor baseline we want to punish.
            worst_ratio = size_ratio ** 2

            if time_ratio <= ideal_ratio:
                scalability_score = 1.0
            elif time_ratio >= worst_ratio:
                scalability_score = 0.0
            else:
                scalability_score = 1.0 - (time_ratio - ideal_ratio) / (worst_ratio - ideal_ratio)
        else:
            scalability_score = 0.5

        # Combined score.
        combined_score = 0.65 * performance_score + 0.35 * scalability_score

        return {
            "metrics": {
                "combined_score": float(combined_score),
                "performance_score": float(performance_score),
                "scalability_score": float(scalability_score),
                "total_time_ms": float(total_time),
            },
            "artifacts": {
                "sizes": sizes,
                "ks": ks,
                "individual_times": times,
                "output": output,
            },
        }

    except subprocess.TimeoutExpired:
        return {
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": "Program timed out (too slow)"},
        }
    except Exception as e:
        return {
            "metrics": {"combined_score": 0.0},
            "artifacts": {"error": str(e)},
        }
