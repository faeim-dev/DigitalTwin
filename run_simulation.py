#!/usr/bin/env python
"""
Main Runner Script
==================
Execute the full digital twin simulation pipeline:
  1. Run baseline and alternative scenarios
  2. Collect metrics
  3. Generate plots and reports

Usage:
    python run_simulation.py
"""

from warehouse_simulator import WarehouseConfig, WarehouseSimulation
from analysis_and_reporting import SimulationReporter
import os


def main():
    """Run the full simulation pipeline."""

    print("\n" + "="*80)
    print("ORDER FULFILLMENT DIGITAL TWIN SIMULATOR")
    print("="*80 + "\n")

    # ========================================================================
    # DEFINE SCENARIOS
    # ========================================================================

    scenarios = {
        'Baseline (Current)': WarehouseConfig(
            num_pickers=3,
            num_qc_stations=2,
            num_packers=2,
            order_arrival_rate=2.5,
        ),

        'Speed (More Resources)': WarehouseConfig(
            num_pickers=4,           # +1 picker
            num_qc_stations=3,       # +1 QC station
            num_packers=3,           # +1 packer
            order_arrival_rate=2.5,
        ),

        'Lean (Fewer Resources)': WarehouseConfig(
            num_pickers=2,           # -1 picker
            num_qc_stations=1,       # -1 QC station
            num_packers=1,           # -1 packer
            order_arrival_rate=2.5,
        ),
    }

    # ========================================================================
    # SIMULATION PARAMETERS
    # ========================================================================

    NUM_ORDERS = 1000          # Total orders to simulate
    SIMULATION_DAYS = 30       # Simulation time horizon
    RANDOM_SEED = 42           # For reproducibility

    # ========================================================================
    # RUN SIMULATIONS
    # ========================================================================

    results = {}

    for scenario_name, config in scenarios.items():
        print(f"\n{'─'*80}")
        print(f"Running: {scenario_name}")
        print(f"{'─'*80}")
        print(f"  Pickers: {config.num_pickers} | QC Stations: {config.num_qc_stations} | Packers: {config.num_packers}")
        print(f"  Arrival Rate: {config.order_arrival_rate} orders/min | QC Failure Rate: {config.qc_failure_rate*100:.1f}%")

        sim = WarehouseSimulation(config, random_seed=RANDOM_SEED)
        metrics = sim.run(num_orders=NUM_ORDERS, simulation_days=SIMULATION_DAYS)

        print(f"\n  Results:")
        print(f"    Orders Completed: {metrics.orders_completed}")
        print(f"    Throughput: {metrics.throughput():.2f} orders/hour")
        print(f"    Avg Fulfillment Time: {metrics.avg_fulfillment_time():.2f} minutes")
        print(f"    Avg Wait Time: {metrics.avg_wait_time():.2f} minutes")
        print(f"    Cost per Order: ${metrics.cost_per_order():.2f}")

        results[scenario_name] = metrics

    # ========================================================================
    # GENERATE REPORTS AND PLOTS
    # ========================================================================

    print(f"\n{'─'*80}")
    print("Generating Reports and Visualizations")
    print(f"{'─'*80}")

    # Create output directory
    output_dir = './simulation_output'
    os.makedirs(output_dir, exist_ok=True)

    # Create reporter
    reporter = SimulationReporter(results, output_dir=output_dir)

    # Generate and save all outputs
    metrics_table = reporter.metrics_table()
    print("\nMetrics Table:")
    print(metrics_table.to_string(index=False))

    # Save metrics table
    metrics_table.to_csv(f'{output_dir}/metrics_comparison.csv', index=False)
    print(f"\n✓ Metrics table saved to: {output_dir}/metrics_comparison.csv")

    # Generate plots
    reporter.save_all_plots(prefix='scenario')

    # Generate text report
    reporter.save_report('simulation_report.txt')
    print(f"\n✓ Text report saved to: {output_dir}/simulation_report.txt")

    # Print full report text to console
    print(f"\n{'='*80}")
    print(reporter.generate_report_text())

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput Directory: {output_dir}/")
    print(f"\nGenerated Files:")
    print(f"  - metrics_comparison.csv")
    print(f"  - scenario_throughput.png")
    print(f"  - scenario_fulfillment_time.png")
    print(f"  - scenario_utilization.png")
    print(f"  - scenario_cost_analysis.png")
    print(f"  - scenario_fulfillment_distribution.png")
    print(f"  - simulation_report.txt")
    print()


if __name__ == '__main__':
    main()
