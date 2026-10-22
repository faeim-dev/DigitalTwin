"""
Order Fulfillment Digital Twin Simulator
========================================
A discrete-event simulation of a warehouse order fulfillment system.
Models order arrivals, picking, quality control, packing, and dispatch.

Usage:
    config = WarehouseConfig(num_pickers=3, num_qc_stations=2, num_packers=2)
    env = WarehouseSimulation(config, random_seed=42)
    metrics = env.run(num_orders=1000, simulation_days=30)
    print(metrics.summary())
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict


@dataclass
class WarehouseConfig:
    """Configuration parameters for warehouse simulation."""
    num_pickers: int = 3
    num_qc_stations: int = 2
    num_packers: int = 2

    # Service time distributions (minutes)
    pick_time_mean: float = 4.0
    pick_time_std: float = 1.5

    qc_time_mean: float = 2.0
    qc_time_std: float = 0.8

    pack_time_mean: float = 3.0
    pack_time_std: float = 1.0

    # Order characteristics
    order_arrival_rate: float = 2.5  # orders per minute
    qc_failure_rate: float = 0.05    # 5% fail QC, need rework
    rework_time_mean: float = 1.5
    rework_time_std: float = 0.5

    # Batch dispatch (every N minutes)
    dispatch_batch_interval: int = 60  # minutes

    # Costs ($/minute)
    picker_hourly_cost: float = 20.0
    qc_hourly_cost: float = 18.0
    packer_hourly_cost: float = 18.0

    def __post_init__(self):
        self.picker_minute_cost = self.picker_hourly_cost / 60
        self.qc_minute_cost = self.qc_hourly_cost / 60
        self.packer_minute_cost = self.packer_hourly_cost / 60


@dataclass
class Order:
    """Represents a single customer order."""
    order_id: int
    arrival_time: float
    creation_time: float = field(default_factory=lambda: None)
    pick_end_time: float = None
    qc_end_time: float = None
    pack_end_time: float = None
    dispatch_time: float = None

    def fulfillment_time(self) -> float:
        """Total time from arrival to dispatch (minutes)."""
        if self.dispatch_time is None:
            return None
        return self.dispatch_time - self.arrival_time

    def wait_before_pick(self) -> float:
        """Wait time in queue before picking."""
        return self.pick_end_time - self.arrival_time - (self.pick_end_time - self.creation_time)

    def __repr__(self):
        return f"Order(id={self.order_id}, arrival={self.arrival_time:.1f})"


@dataclass
class SimulationMetrics:
    """Aggregated simulation results."""
    config: WarehouseConfig
    orders_completed: int = 0
    total_fulfillment_time: float = 0.0
    total_wait_time: float = 0.0
    total_pick_time: float = 0.0
    total_qc_time: float = 0.0
    total_pack_time: float = 0.0

    picker_busy_time: float = 0.0
    qc_busy_time: float = 0.0
    packer_busy_time: float = 0.0

    orders_reworked: int = 0
    simulation_time: float = 0.0

    order_log: List[Order] = field(default_factory=list)

    def avg_fulfillment_time(self) -> float:
        return self.total_fulfillment_time / self.orders_completed if self.orders_completed > 0 else 0

    def avg_wait_time(self) -> float:
        return self.total_wait_time / self.orders_completed if self.orders_completed > 0 else 0

    def throughput(self) -> float:
        """Orders per hour."""
        hours = self.simulation_time / 60
        return self.orders_completed / hours if hours > 0 else 0

    def picker_utilization(self) -> float:
        total_picker_time = self.config.num_pickers * self.simulation_time
        return self.picker_busy_time / total_picker_time if total_picker_time > 0 else 0

    def qc_utilization(self) -> float:
        total_qc_time = self.config.num_qc_stations * self.simulation_time
        return self.qc_busy_time / total_qc_time if total_qc_time > 0 else 0

    def packer_utilization(self) -> float:
        total_packer_time = self.config.num_packers * self.simulation_time
        return self.packer_busy_time / total_packer_time if total_packer_time > 0 else 0

    def total_labor_cost(self) -> float:
        """Total labor cost for the simulation period."""
        picker_cost = self.config.num_pickers * self.simulation_time * self.config.picker_minute_cost
        qc_cost = self.config.num_qc_stations * self.simulation_time * self.config.qc_minute_cost
        packer_cost = self.config.num_packers * self.simulation_time * self.config.packer_minute_cost
        return picker_cost + qc_cost + packer_cost

    def cost_per_order(self) -> float:
        return self.total_labor_cost() / self.orders_completed if self.orders_completed > 0 else 0

    def summary(self) -> str:
        """Return a formatted summary of metrics."""
        lines = [
            f"{'='*60}",
            f"SIMULATION RESULTS",
            f"{'='*60}",
            f"Orders completed:           {self.orders_completed:>10}",
            f"Orders reworked:            {self.orders_reworked:>10}",
            f"Rework rate:                {self.orders_reworked/self.orders_completed*100:>10.2f}%",
            f"",
            f"Throughput (orders/hour):   {self.throughput():>10.2f}",
            f"Avg fulfillment time (min): {self.avg_fulfillment_time():>10.2f}",
            f"Avg wait time (min):        {self.avg_wait_time():>10.2f}",
            f"",
            f"Picker utilization:         {self.picker_utilization()*100:>10.2f}%",
            f"QC utilization:             {self.qc_utilization()*100:>10.2f}%",
            f"Packer utilization:         {self.packer_utilization()*100:>10.2f}%",
            f"",
            f"Total labor cost:           ${self.total_labor_cost():>10,.2f}",
            f"Cost per order:             ${self.cost_per_order():>10.2f}",
            f"Simulation time (hours):    {self.simulation_time/60:>10.2f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class WarehouseSimulation:
    """Main simulation engine for order fulfillment."""

    def __init__(self, config: WarehouseConfig, random_seed: int = 42):
        self.config = config
        self.env = simpy.Environment()
        self.metrics = SimulationMetrics(config=config)

        np.random.seed(random_seed)

        # Create resources (queues with servers)
        self.picker_resource = simpy.Resource(self.env, config.num_pickers)
        self.qc_resource = simpy.Resource(self.env, config.num_qc_stations)
        self.packer_resource = simpy.Resource(self.env, config.num_packers)

        # Tracking
        self.order_counter = 0
        self.dispatch_queue: List[Order] = []

    def order_arrival_generator(self, num_orders: int):
        """Generate order arrivals as a Poisson process."""
        for i in range(num_orders):
            self.order_counter += 1
            order = Order(
                order_id=self.order_counter,
                arrival_time=self.env.now,
                creation_time=self.env.now
            )
            self.env.process(self.process_order(order))

            # Inter-arrival time: exponential with rate = orders/minute
            inter_arrival = np.random.exponential(1 / self.config.order_arrival_rate)
            yield self.env.timeout(inter_arrival)

    def process_order(self, order: Order):
        """Process a single order through all stations."""
        # PICK STATION
        with self.picker_resource.request() as req:
            yield req
            pick_time = max(0.1, np.random.normal(
                self.config.pick_time_mean,
                self.config.pick_time_std
            ))
            yield self.env.timeout(pick_time)
            order.pick_end_time = self.env.now
            self.metrics.total_pick_time += pick_time
            self.metrics.picker_busy_time += pick_time

        # QC STATION (with rework loop)
        qc_passed = False
        while not qc_passed:
            with self.qc_resource.request() as req:
                yield req
                qc_time = max(0.1, np.random.normal(
                    self.config.qc_time_mean,
                    self.config.qc_time_std
                ))
                yield self.env.timeout(qc_time)
                self.metrics.total_qc_time += qc_time
                self.metrics.qc_busy_time += qc_time

                # Check if passes QC
                if np.random.random() > self.config.qc_failure_rate:
                    qc_passed = True
                    order.qc_end_time = self.env.now
                else:
                    self.metrics.orders_reworked += 1
                    rework_time = max(0.1, np.random.normal(
                        self.config.rework_time_mean,
                        self.config.rework_time_std
                    ))
                    yield self.env.timeout(rework_time)

        # PACK STATION
        with self.packer_resource.request() as req:
            yield req
            pack_time = max(0.1, np.random.normal(
                self.config.pack_time_mean,
                self.config.pack_time_std
            ))
            yield self.env.timeout(pack_time)
            order.pack_end_time = self.env.now
            self.metrics.total_pack_time += pack_time
            self.metrics.packer_busy_time += pack_time

        # Add to dispatch queue (will be picked up by batch dispatcher)
        self.dispatch_queue.append(order)

    def dispatch_batches(self, simulation_end_time: float):
        """Periodically dispatch batches of orders."""
        while self.env.now < simulation_end_time:
            yield self.env.timeout(self.config.dispatch_batch_interval)

            # Dispatch all pending orders
            for order in self.dispatch_queue:
                order.dispatch_time = self.env.now
                self.metrics.order_log.append(order)
                self.metrics.orders_completed += 1
                self.metrics.total_fulfillment_time += order.fulfillment_time()
                self.metrics.total_wait_time += order.fulfillment_time() - (
                    self.config.pick_time_mean + self.config.qc_time_mean + self.config.pack_time_mean
                )

            self.dispatch_queue.clear()

    def run(self, num_orders: int = 1000, simulation_days: int = 30) -> SimulationMetrics:
        """Run the full simulation."""
        simulation_end_time = simulation_days * 24 * 60  # Convert days to minutes

        # Start all processes
        self.env.process(self.order_arrival_generator(num_orders))
        self.env.process(self.dispatch_batches(simulation_end_time))

        # Run simulation
        self.env.run(until=simulation_end_time)

        # Finalize metrics
        self.metrics.simulation_time = self.env.now

        # Dispatch any remaining orders
        for order in self.dispatch_queue:
            if order.dispatch_time is None:
                order.dispatch_time = self.env.now
                self.metrics.order_log.append(order)
                self.metrics.orders_completed += 1
                if order.fulfillment_time() is not None:
                    self.metrics.total_fulfillment_time += order.fulfillment_time()

        return self.metrics


# ============================================================================
# HELPER FUNCTIONS FOR SCENARIO ANALYSIS
# ============================================================================

def run_scenario(name: str, config: WarehouseConfig, num_orders: int = 1000,
                 simulation_days: int = 30, random_seed: int = 42) -> SimulationMetrics:
    """Run a single simulation scenario and return metrics."""
    print(f"\nRunning scenario: {name}")
    print(f"  Config: {config.num_pickers} pickers, {config.num_qc_stations} QC, {config.num_packers} packers")

    sim = WarehouseSimulation(config, random_seed=random_seed)
    metrics = sim.run(num_orders=num_orders, simulation_days=simulation_days)

    return metrics


def compare_scenarios(scenarios: Dict[str, WarehouseConfig], num_orders: int = 1000,
                     simulation_days: int = 30, random_seed: int = 42) -> pd.DataFrame:
    """Run multiple scenarios and return comparison dataframe."""
    results = []

    for scenario_name, config in scenarios.items():
        metrics = run_scenario(scenario_name, config, num_orders, simulation_days, random_seed)

        results.append({
            'Scenario': scenario_name,
            'Throughput (orders/hr)': metrics.throughput(),
            'Avg Fulfillment Time (min)': metrics.avg_fulfillment_time(),
            'Avg Wait Time (min)': metrics.avg_wait_time(),
            'Picker Utilization (%)': metrics.picker_utilization() * 100,
            'QC Utilization (%)': metrics.qc_utilization() * 100,
            'Packer Utilization (%)': metrics.packer_utilization() * 100,
            'Total Labor Cost ($)': metrics.total_labor_cost(),
            'Cost per Order ($)': metrics.cost_per_order(),
            'Orders Reworked': metrics.orders_reworked,
            'Rework Rate (%)': (metrics.orders_reworked / metrics.orders_completed) * 100,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Define three policy scenarios
    scenarios = {
        'Baseline (Current)': WarehouseConfig(
            num_pickers=3,
            num_qc_stations=2,
            num_packers=2
        ),
        'Speed (More Resources)': WarehouseConfig(
            num_pickers=4,
            num_qc_stations=3,
            num_packers=3
        ),
        'Lean (Fewer Resources)': WarehouseConfig(
            num_pickers=2,
            num_qc_stations=1,
            num_packers=1
        ),
    }

    # Run comparison
    comparison = compare_scenarios(scenarios, num_orders=1000, simulation_days=30)

    print("\n" + "="*100)
    print("SCENARIO COMPARISON")
    print("="*100)
    print(comparison.to_string(index=False))
    print("\n")

    # Save results
    comparison.to_csv('/private/tmp/claude-503/-Users-faghda3/0f30fb17-85e8-4ac8-a15c-bee1278f397c/scratchpad/simulation_results.csv', index=False)
    print("Results saved to: simulation_results.csv")
