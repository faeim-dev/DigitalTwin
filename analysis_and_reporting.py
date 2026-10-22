"""
Analysis and Reporting Module
==============================
Generates plots, tables, and report outputs from simulation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warehouse_simulator import SimulationMetrics
from typing import Dict, List


class SimulationReporter:
    """Generates reports, plots, and statistics from simulation metrics."""

    def __init__(self, results: Dict[str, SimulationMetrics], output_dir: str = '.'):
        """
        Args:
            results: Dict mapping scenario names to SimulationMetrics objects
            output_dir: Directory to save plots and reports
        """
        self.results = results
        self.output_dir = output_dir

    def metrics_table(self) -> pd.DataFrame:
        """Create a comprehensive metrics table."""
        rows = []

        for scenario_name, metrics in self.results.items():
            rows.append({
                'Scenario': scenario_name,
                'Orders Completed': metrics.orders_completed,
                'Throughput (orders/hr)': f"{metrics.throughput():.2f}",
                'Avg Fulfillment (min)': f"{metrics.avg_fulfillment_time():.2f}",
                'Avg Wait (min)': f"{metrics.avg_wait_time():.2f}",
                'Picker Util. (%)': f"{metrics.picker_utilization()*100:.1f}",
                'QC Util. (%)': f"{metrics.qc_utilization()*100:.1f}",
                'Packer Util. (%)': f"{metrics.packer_utilization()*100:.1f}",
                'Total Labor Cost ($)': f"${metrics.total_labor_cost():.0f}",
                'Cost/Order ($)': f"${metrics.cost_per_order():.2f}",
                'Rework Rate (%)': f"{(metrics.orders_reworked/metrics.orders_completed)*100:.2f}",
            })

        return pd.DataFrame(rows)

    def plot_throughput_comparison(self, figsize=(10, 6)):
        """Plot throughput across scenarios."""
        scenarios = list(self.results.keys())
        throughputs = [self.results[s].throughput() for s in scenarios]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(scenarios, throughputs, color=['#3498db', '#e74c3c', '#2ecc71'])

        ax.set_ylabel('Throughput (orders/hour)', fontsize=12)
        ax.set_title('Order Throughput by Scenario', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        return fig

    def plot_fulfillment_time(self, figsize=(10, 6)):
        """Plot average fulfillment time across scenarios."""
        scenarios = list(self.results.keys())
        times = [self.results[s].avg_fulfillment_time() for s in scenarios]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(scenarios, times, color=['#3498db', '#e74c3c', '#2ecc71'])

        ax.set_ylabel('Average Fulfillment Time (minutes)', fontsize=12)
        ax.set_title('Order Fulfillment Time by Scenario', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        return fig

    def plot_utilization_heatmap(self, figsize=(10, 6)):
        """Plot worker utilization across scenarios and stations."""
        scenarios = list(self.results.keys())

        utilization_data = []
        for scenario in scenarios:
            metrics = self.results[scenario]
            utilization_data.append([
                metrics.picker_utilization() * 100,
                metrics.qc_utilization() * 100,
                metrics.packer_utilization() * 100,
            ])

        utilization_df = pd.DataFrame(
            utilization_data,
            index=scenarios,
            columns=['Pickers', 'QC Stations', 'Packers']
        )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(utilization_df.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(scenarios)
        ax.set_yticklabels(utilization_df.columns)

        # Add text annotations
        for i in range(3):
            for j in range(len(scenarios)):
                text = ax.text(j, i, f'{utilization_df.iloc[j, i]:.0f}%',
                             ha="center", va="center", color="black", fontsize=11)

        ax.set_title('Worker Utilization (%) by Station and Scenario', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Utilization (%)')
        plt.tight_layout()
        return fig

    def plot_cost_analysis(self, figsize=(12, 6)):
        """Plot cost vs. performance trade-off."""
        scenarios = list(self.results.keys())
        costs = [self.results[s].cost_per_order() for s in scenarios]
        throughputs = [self.results[s].throughput() for s in scenarios]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Cost per order
        bars1 = ax1.bar(scenarios, costs, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax1.set_ylabel('Cost per Order ($)', fontsize=12)
        ax1.set_title('Labor Cost per Order', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom', fontsize=10)

        # Throughput
        bars2 = ax2.bar(scenarios, throughputs, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_ylabel('Throughput (orders/hour)', fontsize=12)
        ax2.set_title('System Throughput', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        return fig

    def plot_fulfillment_time_distribution(self, figsize=(12, 6)):
        """Plot distribution of fulfillment times by scenario."""
        fig, axes = plt.subplots(1, len(self.results), figsize=figsize)
        if len(self.results) == 1:
            axes = [axes]

        for ax, (scenario_name, metrics) in zip(axes, self.results.items()):
            fulfillment_times = [
                order.fulfillment_time() for order in metrics.order_log
                if order.fulfillment_time() is not None
            ]

            ax.hist(fulfillment_times, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Fulfillment Time (minutes)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{scenario_name}\n(μ={np.mean(fulfillment_times):.1f} min)',
                        fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_report_text(self) -> str:
        """Generate a text report suitable for inclusion in a paper."""
        lines = []

        lines.append("="*80)
        lines.append("SIMULATION RESULTS REPORT")
        lines.append("="*80)
        lines.append("")

        # Summary table
        lines.append("TABLE 1: SCENARIO COMPARISON")
        lines.append("-"*80)
        lines.append(self.metrics_table().to_string(index=False))
        lines.append("")

        # Interpretation
        lines.append("INTERPRETATION OF RESULTS")
        lines.append("-"*80)

        baseline_metrics = self.results['Baseline (Current)']
        speed_metrics = self.results['Speed (More Resources)']
        lean_metrics = self.results['Lean (Fewer Resources)']

        baseline_throughput = baseline_metrics.throughput()
        speed_throughput = speed_metrics.throughput()
        lean_throughput = lean_metrics.throughput()

        baseline_cost = baseline_metrics.cost_per_order()
        speed_cost = speed_metrics.cost_per_order()
        lean_cost = lean_metrics.cost_per_order()

        lines.append(f"\nThroughput Analysis:")
        lines.append(f"  - Speed scenario increases throughput by {(speed_throughput/baseline_throughput - 1)*100:.1f}%")
        lines.append(f"  - Lean scenario reduces throughput by {(1 - lean_throughput/baseline_throughput)*100:.1f}%")

        lines.append(f"\nCost Analysis:")
        lines.append(f"  - Speed scenario increases cost per order by ${speed_cost - baseline_cost:.2f} ({(speed_cost/baseline_cost - 1)*100:.1f}%)")
        lines.append(f"  - Lean scenario decreases cost per order by ${baseline_cost - lean_cost:.2f} ({(1 - lean_cost/baseline_cost)*100:.1f}%)")

        lines.append(f"\nOperational Efficiency:")
        lines.append(f"  - Baseline picker utilization: {baseline_metrics.picker_utilization()*100:.1f}%")
        lines.append(f"  - Speed picker utilization: {speed_metrics.picker_utilization()*100:.1f}%")
        lines.append(f"  - Lean picker utilization: {lean_metrics.picker_utilization()*100:.1f}%")

        lines.append("")
        lines.append("="*80)

        return "\n".join(lines)

    def save_all_plots(self, prefix: str = 'simulation'):
        """Save all plots to files."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Saving plots to {self.output_dir}/")

        fig1 = self.plot_throughput_comparison()
        fig1.savefig(f'{self.output_dir}/{prefix}_throughput.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ {prefix}_throughput.png")
        plt.close(fig1)

        fig2 = self.plot_fulfillment_time()
        fig2.savefig(f'{self.output_dir}/{prefix}_fulfillment_time.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ {prefix}_fulfillment_time.png")
        plt.close(fig2)

        fig3 = self.plot_utilization_heatmap()
        fig3.savefig(f'{self.output_dir}/{prefix}_utilization.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ {prefix}_utilization.png")
        plt.close(fig3)

        fig4 = self.plot_cost_analysis()
        fig4.savefig(f'{self.output_dir}/{prefix}_cost_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ {prefix}_cost_analysis.png")
        plt.close(fig4)

        fig5 = self.plot_fulfillment_time_distribution()
        fig5.savefig(f'{self.output_dir}/{prefix}_fulfillment_distribution.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ {prefix}_fulfillment_distribution.png")
        plt.close(fig5)

    def save_report(self, filename: str = 'simulation_report.txt'):
        """Save text report to file."""
        report = self.generate_report_text()

        with open(f'{self.output_dir}/{filename}', 'w') as f:
            f.write(report)

        print(f"Report saved to {self.output_dir}/{filename}")
