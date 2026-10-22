import math
import os

import pandas as pd

from warehouse_simulator import WarehouseConfig, WarehouseSimulation
from analysis_and_reporting import SimulationReporter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, 'DataCoSupplyChainDataset.csv')
OUTPUT_DIR = os.path.join(HERE, 'dataco_output')
RANDOM_SEED = 42
QC_FAILURE_RATE = 0.05


def load_orders():
    df = pd.read_csv(DATA_PATH, encoding='latin-1', low_memory=False)
    df['order_time'] = pd.to_datetime(df['order date (DateOrders)'], format='%m/%d/%Y %H:%M')
    df['ship_time'] = pd.to_datetime(df['shipping date (DateOrders)'], format='%m/%d/%Y %H:%M')
    orders = df.groupby('Order Id').agg(
        order_time=('order_time', 'first'),
        ship_time=('ship_time', 'first'),
        delivery_status=('Delivery Status', 'first'),
    )
    print(f"Line items: {len(df):,}  Unique orders: {len(orders):,}")
    canceled = orders['delivery_status'] == 'Shipping canceled'
    print(f"Excluding 'Shipping canceled': {canceled.sum():,}")
    return orders[~canceled].sort_values('order_time')


def calibrate(orders):
    times = orders['order_time']
    span_min = (times.iloc[-1] - times.iloc[0]).total_seconds() / 60
    rate = len(orders) / span_min
    gaps = times.diff().dropna().dt.total_seconds() / 60
    cv = gaps.std() / gaps.mean()

    days = times.dt.normalize().nunique()
    per_hour = times.dt.hour.value_counts().sort_index() / days

    lead_h = (orders['ship_time'] - orders['order_time']).dt.total_seconds() / 3600

    print("\nCalibration from data")
    print(f"  Period: {times.iloc[0]} to {times.iloc[-1]} ({span_min / 1440:.0f} days)")
    print(f"  Arrival rate: {rate:.5f} orders/min ({rate * 60:.2f} orders/hour)")
    print(f"  Inter-arrival CV: {cv:.3f} (exponential = 1.0)")
    print(f"  Orders per hour of day: min {per_hour.min():.2f}, max {per_hour.max():.2f}")
    print(f"  Order-to-ship lead time (reference only): median {lead_h.median():.1f} h, "
          f"IQR {lead_h.quantile(0.25):.1f}-{lead_h.quantile(0.75):.1f} h")

    return rate, span_min


def build_scenarios(rate):
    common = dict(order_arrival_rate=rate, qc_failure_rate=QC_FAILURE_RATE)
    return {
        'Baseline (Current)': WarehouseConfig(num_pickers=3, num_qc_stations=2, num_packers=2, **common),
        'Speed (More Resources)': WarehouseConfig(num_pickers=4, num_qc_stations=3, num_packers=3, **common),
        'Lean (Fewer Resources)': WarehouseConfig(num_pickers=2, num_qc_stations=1, num_packers=1, **common),
    }


def main():
    orders = load_orders()
    rate, _ = calibrate(orders)

    num_orders = len(orders)
    simulation_days = math.ceil(num_orders / rate / 1440 * 1.05) + 2
    print(f"\nSimulating {num_orders:,} orders over {simulation_days} simulated days\n")

    results = {}
    for name, config in build_scenarios(rate).items():
        sim = WarehouseSimulation(config, random_seed=RANDOM_SEED)
        metrics = sim.run(num_orders=num_orders, simulation_days=simulation_days)
        results[name] = metrics
        print(f"{name}: {metrics.orders_completed:,} completed, "
              f"throughput {metrics.throughput():.2f}/h, "
              f"picker util {metrics.picker_utilization() * 100:.2f}%, "
              f"cost/order ${metrics.cost_per_order():.2f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reporter = SimulationReporter(results, output_dir=OUTPUT_DIR)
    table = reporter.metrics_table()
    table.to_csv(os.path.join(OUTPUT_DIR, 'metrics_comparison.csv'), index=False)
    reporter.save_all_plots(prefix='dataco')
    reporter.save_report('dataco_report.txt')
    print("\n" + table.to_string(index=False))
    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
