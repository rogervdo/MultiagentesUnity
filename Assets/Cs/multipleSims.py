"""
Test script for comparative heuristics vs no heuristics analysis
"""

import sys

sys.path.append("/Users/roger/Code/Clases/Multiagentes/ipynb")

from controller import run_multiple_simulations_and_plot


def test_comparative_analysis():
    """Test the comparative heuristics vs no heuristics functionality"""
    print("Testing comparative heuristics vs no heuristics analysis...")

    # Run comparative analysis with 3 runs and 50 steps each for quick testing
    results = run_multiple_simulations_and_plot(num_runs=5, steps=300)

    print("\nComparative analysis completed successfully!")
    print("Results keys:", list(results.keys()))

    # Print some sample comparative statistics
    h_active = results["heuristics"]["avg_active_cars"]
    nh_active = results["no_heuristics"]["avg_active_cars"]
    h_arrivals = results["heuristics"]["avg_cumulative_arrivals"]
    nh_arrivals = results["no_heuristics"]["avg_cumulative_arrivals"]
    h_waiting = results["heuristics"]["avg_waiting_steps"]
    nh_waiting = results["no_heuristics"]["avg_waiting_steps"]

    print("\nFinal Results:")
    print(f"Heuristics - Avg Active Cars: {h_active[-1]:.2f}")
    print(f"No Heuristics - Avg Active Cars: {nh_active[-1]:.2f}")
    print(f"Heuristics - Avg Cumulative Arrivals: {h_arrivals[-1]:.2f}")
    print(f"No Heuristics - Avg Cumulative Arrivals: {nh_arrivals[-1]:.2f}")
    print(f"Heuristics - Avg Waiting Steps: {h_waiting[-1]:.2f}")
    print(f"No Heuristics - Avg Waiting Steps: {nh_waiting[-1]:.2f}")
    print(f"Active Cars Difference (H - NH): {h_active[-1] - nh_active[-1]:.2f}")
    print(
        f"Cumulative Arrivals Difference (H - NH): {h_arrivals[-1] - nh_arrivals[-1]:.2f}"
    )
    print(
        f"Average Waiting Steps Difference (H - NH): {h_waiting[-1] - nh_waiting[-1]:.2f}"
    )


if __name__ == "__main__":
    test_comparative_analysis()