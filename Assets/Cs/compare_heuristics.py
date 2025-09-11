"""
Traffic Simulation Comparison: Heuristics vs No Heuristics

This script runs traffic simulations with and without heuristics-based traffic light control
and compares their performance metrics.

Usage:
    python compare_heuristics.py

The script will:
1. Run simulation without heuristics
2. Run simulation with heuristics
3. Compare and display results
4. Generate comparison plots
5. Save animations as 'Traffic_heuristics.gif' and 'Traffic_heuristicsno.gif' (if enabled)
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import time
from datetime import datetime

# Import the traffic simulation modules
import reto6 as traffic_sim


def run_simulation_with_heuristics(
    steps=100,
    preset="morning",
    use_heuristics=True,
    enable_animation=True,
    animation_filename=None,
):
    """Run a traffic simulation with or without heuristics"""
    print(f"\n{'=' * 50}")
    print(
        f"Running simulation with {'heuristics' if use_heuristics else 'no heuristics'}"
    )
    print(f"Steps: {steps}, Preset: {preset}")
    print(f"{'=' * 50}")

    # Create model with specified heuristics setting
    if use_heuristics and traffic_sim.HEURISTICS_AVAILABLE:
        model = traffic_sim.create_traffic_model_class(use_heuristics=True)(
            traffic_sim.params
        )
    else:
        model = traffic_sim.create_traffic_model_class(use_heuristics=False)(
            traffic_sim.params
        )

    # Override parameters
    model.p["steps"] = steps
    model.p["preset"] = preset
    model.p["animation"] = (
        enable_animation  # Enable/disable animation based on parameter
    )

    model.setup()

    # Track metrics over time
    metrics = {
        "time": [],
        "active_cars": [],
        "total_spawned": [],
        "total_arrived": [],
        "active_group": [],
        "timer_value": [],
        "group_car_counts": {0: [], 1: [], 2: []},
        "group_densities": {0: [], 1: [], 2: []},
        "group_node_counts": {
            0: len(model.group_nodes[0]),
            1: len(model.group_nodes[1]),
            2: len(model.group_nodes[2]),
        },
    }

    start_time = time.time()

    for t in range(steps):
        model.t = t + 1
        model.step()

        # Record metrics
        metrics["time"].append(model.t)
        metrics["active_cars"].append(len(model.cars))
        metrics["total_spawned"].append(model.total_cars_spawned)
        metrics["total_arrived"].append(model.total_cars_arrived)
        metrics["active_group"].append(model.active_group)
        metrics["timer_value"].append(model.group_cycle_steps)

        # Record car counts per group (if heuristics available)
        if hasattr(model, "get_group_car_count"):
            for group in range(3):
                car_count = model.get_group_car_count(group)
                metrics["group_car_counts"][group].append(car_count)
                if hasattr(model, "get_group_density"):
                    density = model.get_group_density(group)
                    metrics["group_densities"][group].append(density)
                else:
                    metrics["group_densities"][group].append(0)
        else:
            for group in range(3):
                metrics["group_car_counts"][group].append(0)
                metrics["group_densities"][group].append(0)

    end_time = time.time()
    simulation_time = end_time - start_time

    print(".2f")
    print("Final Statistics:")
    print(f"  Total cars spawned: {model.total_cars_spawned}")
    print(f"  Total cars arrived: {model.total_cars_arrived}")
    print(f"  Cars still active: {len(model.cars)}")
    print(f"  Average cars per step: {np.mean(metrics['active_cars']):.1f}")

    return metrics, simulation_time


def compare_results(metrics_no_heuristics, metrics_heuristics, time_no_h, time_h):
    """Compare and display results from both simulations"""
    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS")
    print(f"{'=' * 60}")

    # Calculate key metrics
    avg_cars_no_h = np.mean(metrics_no_heuristics["active_cars"])
    avg_cars_h = np.mean(metrics_heuristics["active_cars"])

    total_arrived_no_h = metrics_no_heuristics["total_arrived"][-1]
    total_arrived_h = metrics_heuristics["total_arrived"][-1]

    total_spawned_no_h = metrics_no_heuristics["total_spawned"][-1]
    total_spawned_h = metrics_heuristics["total_spawned"][-1]

    # Calculate efficiency (arrived / spawned)
    efficiency_no_h = total_arrived_no_h / max(total_spawned_no_h, 1) * 100
    efficiency_h = total_arrived_h / max(total_spawned_h, 1) * 100

    print("Performance Metrics:")
    print(f"  Average active cars - No heuristics: {avg_cars_no_h:.1f}")
    print(f"  Average active cars - With heuristics: {avg_cars_h:.1f}")
    print(
        f"  Difference: {avg_cars_h - avg_cars_no_h:+.1f} ({((avg_cars_h / avg_cars_no_h - 1) * 100):+.1f}%)"
    )

    print(f"\n  Total cars arrived - No heuristics: {total_arrived_no_h}")
    print(f"  Total cars arrived - With heuristics: {total_arrived_h}")
    print(f"  Difference: {total_arrived_h - total_arrived_no_h:+d}")

    print(f"\n  Efficiency (arrived/spawned) - No heuristics: {efficiency_no_h:.1f}%")
    print(f"  Efficiency (arrived/spawned) - With heuristics: {efficiency_h:.1f}%")
    print(f"  Difference: {efficiency_h - efficiency_no_h:+.1f}%")

    print(f"\n  Simulation time - No heuristics: {time_no_h:.2f}s")
    print(f"  Simulation time - With heuristics: {time_h:.2f}s")
    print(f"  Time difference: {time_h - time_no_h:+.2f}s")

    # Timer analysis (only for heuristics)
    if traffic_sim.HEURISTICS_AVAILABLE:
        timer_values = metrics_heuristics["timer_value"]
        avg_timer = np.mean(timer_values)
        min_timer = min(timer_values)
        max_timer = max(timer_values)

        print("\n  Timer Analysis (Heuristics):")
        print(f"    Average timer: {avg_timer:.1f}")
        print(f"    Min timer: {min_timer}")
        print(f"    Max timer: {max_timer}")
        print(f"    Timer adjustments: {len(set(timer_values))} different values")

        # Density analysis
        print("\n  Density Analysis (Heuristics):")
        for group in range(3):
            node_count = metrics_heuristics["group_node_counts"][group]
            avg_density = np.mean(metrics_heuristics["group_densities"][group])
            print(
                f"    Group {group}: {node_count} nodes, avg density: {avg_density:.3f} cars/node"
            )

    return {
        "avg_cars_no_h": avg_cars_no_h,
        "avg_cars_h": avg_cars_h,
        "total_arrived_no_h": total_arrived_no_h,
        "total_arrived_h": total_arrived_h,
        "efficiency_no_h": efficiency_no_h,
        "efficiency_h": efficiency_h,
        "time_no_h": time_no_h,
        "time_h": time_h,
    }


def plot_comparison(metrics_no_heuristics, metrics_heuristics):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Traffic Simulation: Heuristics vs No Heuristics Comparison", fontsize=16
    )

    # Plot 1: Active cars over time
    axes[0, 0].plot(
        metrics_no_heuristics["time"],
        metrics_no_heuristics["active_cars"],
        label="No Heuristics",
        color="blue",
        alpha=0.7,
    )
    axes[0, 0].plot(
        metrics_heuristics["time"],
        metrics_heuristics["active_cars"],
        label="With Heuristics",
        color="red",
        alpha=0.7,
    )
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Active Cars")
    axes[0, 0].set_title("Active Cars Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Cumulative arrivals
    axes[0, 1].plot(
        metrics_no_heuristics["time"],
        metrics_no_heuristics["total_arrived"],
        label="No Heuristics",
        color="blue",
        alpha=0.7,
    )
    axes[0, 1].plot(
        metrics_heuristics["time"],
        metrics_heuristics["total_arrived"],
        label="With Heuristics",
        color="red",
        alpha=0.7,
    )
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Total Cars Arrived")
    axes[0, 1].set_title("Cumulative Car Arrivals")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Traffic light groups
    axes[1, 0].plot(
        metrics_no_heuristics["time"],
        metrics_no_heuristics["active_group"],
        label="No Heuristics",
        color="blue",
        alpha=0.7,
        drawstyle="steps-post",
    )
    axes[1, 0].plot(
        metrics_heuristics["time"],
        metrics_heuristics["active_group"],
        label="With Heuristics",
        color="red",
        alpha=0.7,
        drawstyle="steps-post",
    )
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Active Group")
    axes[1, 0].set_title("Traffic Light Group Changes")
    axes[1, 0].set_yticks([0, 1, 2])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Timer values (only if heuristics available)
    if traffic_sim.HEURISTICS_AVAILABLE:
        axes[1, 1].plot(
            metrics_heuristics["time"],
            metrics_heuristics["timer_value"],
            label="Timer Value",
            color="green",
            alpha=0.7,
            drawstyle="steps-post",
        )
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("Timer Value")
        axes[1, 1].set_title("Adaptive Timer Values")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Heuristics not available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Timer Values (N/A)")

    plt.tight_layout()
    plt.savefig("traffic_comparison_results.png", dpi=300, bbox_inches="tight")
    print("\nComparison plot saved as 'traffic_comparison_results.png'")
    plt.show()


def save_comparison_data(metrics_no_heuristics, metrics_heuristics, comparison_results):
    """Save comparison data to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "timestamp": timestamp,
        "heuristics_available": traffic_sim.HEURISTICS_AVAILABLE,
        "comparison_results": comparison_results,
        "metrics_no_heuristics": metrics_no_heuristics,
        "metrics_heuristics": metrics_heuristics,
    }

    filename = f"traffic_comparison_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDetailed results saved to '{filename}'")


def main():
    """Main comparison function"""
    print("Traffic Simulation Heuristics Comparison")
    print("=" * 50)

    # Check if heuristics are available
    if traffic_sim.HEURISTICS_AVAILABLE:
        print("✓ Heuristics module available")
    else:
        print("✗ Heuristics module not available - running basic comparison only")

    # Simulation parameters
    steps = 300  # Shorter for testing, increase for more comprehensive results
    preset = "morning"

    # Run simulations
    print(f"\nRunning simulations with {steps} steps using '{preset}' preset...")

    # Check if animations should be enabled
    enable_animation = traffic_sim.params.get("animation", True)
    if enable_animation:
        print("✓ Animations enabled")
    else:
        print("✗ Animations disabled")

    # Run without heuristics
    metrics_no_h, time_no_h = run_simulation_with_heuristics(
        steps, preset, use_heuristics=False, enable_animation=enable_animation
    )

    # Save animation for no heuristics if enabled
    if enable_animation:
        # Temporarily modify params for animation
        original_params = traffic_sim.params.copy()
        traffic_sim.params.update({"steps": steps, "preset": preset, "animation": True})

        # Create and save animation using reto6's function
        model, anim = traffic_sim.run_simulation_with_animation(
            "Traffic_heuristicsno.gif"
        )
        print("Animation saved as 'Traffic_heuristicsno.gif'")

        # Restore original params
        traffic_sim.params.update(original_params)

    # Run with heuristics (if available)
    if traffic_sim.HEURISTICS_AVAILABLE:
        metrics_h, time_h = run_simulation_with_heuristics(
            steps, preset, use_heuristics=True, enable_animation=enable_animation
        )

        # Save animation for heuristics if enabled
        if enable_animation:
            # Temporarily modify params for animation
            original_params = traffic_sim.params.copy()
            traffic_sim.params.update(
                {"steps": steps, "preset": preset, "animation": True}
            )

            # Create and save animation using reto6's function
            model, anim = traffic_sim.run_simulation_with_animation(
                "Traffic_heuristics.gif"
            )
            print("Animation saved as 'Traffic_heuristics.gif'")

            # Restore original params
            traffic_sim.params.update(original_params)
    else:
        print("\nSkipping heuristics simulation (module not available)")
        metrics_h, time_h = metrics_no_h, time_no_h  # Use same data for comparison

    # Compare results
    comparison_results = compare_results(metrics_no_h, metrics_h, time_no_h, time_h)

    # Create plots
    plot_comparison(metrics_no_h, metrics_h)

    # Save detailed data
    save_comparison_data(metrics_no_h, metrics_h, comparison_results)

    print(f"\n{'=' * 60}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()