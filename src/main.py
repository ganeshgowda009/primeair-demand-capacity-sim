import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_demand_data(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    months = pd.date_range(start="2024-01-01", periods=12, freq="M")
    demand = np.random.randint(800, 1200, size=12)

    df = pd.DataFrame({"Month": months, "Actual_Demand": demand})
    df["Forecast_Rolling_3M"] = df["Actual_Demand"].rolling(window=3).mean()
    return df

def simulate_inventory(df: pd.DataFrame, starting_inventory: int = 3000, inbound_supply: int = 1000) -> pd.DataFrame:
    inventory = []
    current_inventory = starting_inventory

    for d in df["Actual_Demand"]:
        ending_inventory = current_inventory + inbound_supply - d
        inventory.append(ending_inventory)
        current_inventory = ending_inventory

    df["Ending_Inventory"] = inventory
    df["MOS"] = df["Ending_Inventory"] / df["Forecast_Rolling_3M"]
    return df

def simulate_capacity(df: pd.DataFrame, daily_capacity: int = 1000, spike_pct: float = 0.20) -> pd.DataFrame:
    df["Utilization_%"] = (df["Actual_Demand"] / daily_capacity) * 100
    df["Spike_Demand"] = df["Actual_Demand"] * (1 + spike_pct)
    df["Spike_Utilization_%"] = (df["Spike_Demand"] / daily_capacity) * 100
    return df

def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["Month"], df["Actual_Demand"], label="Actual Demand")
    plt.plot(df["Month"], df["Forecast_Rolling_3M"], label="Rolling 3M Forecast")
    plt.title("Demand vs Forecast")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_dir / "demand_vs_forecast.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["Month"], df["Utilization_%"], label="Utilization % (Baseline)")
    plt.plot(df["Month"], df["Spike_Utilization_%"], label="Utilization % (+20% Demand Spike)")
    plt.title("Capacity Utilization: Baseline vs Demand Spike")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_dir / "capacity_utilization.png")
    plt.close()

def main() -> None:
    df = generate_demand_data()
    df = simulate_inventory(df)
    df = simulate_capacity(df)

    print("\n=== Preview ===")
    print(df.to_string(index=False))

    save_plots(df, Path("outputs"))
    df.to_csv("outputs/results.csv", index=False)

    print("\nSaved outputs to /outputs (plots + results.csv)")

if __name__ == "__main__":
    main()
