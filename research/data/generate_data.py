"""
Synthetic battery charging-cycle data generator.

Matches the schema:
Cycle_Number, Cycle_Type, Time, Ambient_Temperature, Delta t, Current_measured,
Voltage_measured, Current_Mod, cummulative Ah, Ah, soc

Simulates N "vehicles" (fleets), each with its own charging behavior and
degradation trajectory, so we have sparse, heterogeneous data to feed the
federated learning layer -- this mirrors the real-world situation where each
EV's BMS only ever sees its own cycles.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

def simulate_vehicle(vehicle_id, n_cycles=120, base_capacity=2.2,
                      degradation_rate=None, temp_bias=None, fast_charge_bias=None):
    """
    Simulate one vehicle's charging-cycle history.
    Each vehicle has its own latent 'usage personality' (temperature exposure,
    fast-charging habits) which drives a different degradation curve -- this
    is what makes the federated setting non-IID and realistic.
    """
    if degradation_rate is None:
        degradation_rate = RNG.uniform(0.0006, 0.0018)   # per-cycle capacity fade
    if temp_bias is None:
        temp_bias = RNG.uniform(-3, 10)                   # some vehicles run hotter
    if fast_charge_bias is None:
        fast_charge_bias = RNG.uniform(0.1, 0.6)           # fraction of fast-charge cycles

    rows = []
    cumulative_ah = 0.0
    soh = 1.0  # State of Health, starts at 100%

    for cycle in range(1, n_cycles + 1):
        is_fast = RNG.random() < fast_charge_bias
        cycle_type = "fast_charge" if is_fast else "normal_charge"

        ambient_temp = RNG.normal(25 + temp_bias, 3)
        # fast charging + heat accelerates degradation (nonlinear, realistic-ish)
        stress = (1.8 if is_fast else 1.0) * (1 + max(0, ambient_temp - 30) * 0.05)
        soh -= degradation_rate * stress * RNG.uniform(0.85, 1.15)
        soh = max(soh, 0.55)  # floor so it doesn't degrade below realistic EOL

        capacity_now = base_capacity * soh
        # sparse sampling within a cycle: only a handful of timestamped readings
        n_samples = RNG.integers(3, 8)
        t = 0.0
        for s in range(n_samples):
            dt = RNG.uniform(20, 90)  # seconds between sparse readings
            t += dt
            current = RNG.uniform(1.5, 4.0) * (1.6 if is_fast else 1.0) * RNG.choice([-1, 1], p=[0.4, 0.6])
            voltage = RNG.uniform(3.4, 4.2) - (0.15 * (1 - soh))
            current_mod = abs(current) * RNG.uniform(0.95, 1.05)
            ah = current * (dt / 3600.0)
            cumulative_ah += ah
            soc = float(np.clip(RNG.uniform(0.2, 1.0) * soh, 0, 1))

            rows.append({
                "Vehicle_ID": f"V{vehicle_id}",
                "Cycle_Number": cycle,
                "Cycle_Type": cycle_type,
                "Time": round(t, 2),
                "Ambient_Temperature": round(ambient_temp, 2),
                "Delta t ": round(dt, 2),
                "Current_measured": round(current, 4),
                "Voltage_measured": round(voltage, 4),
                "Current_Mod": round(current_mod, 4),
                "cummulative Ah": round(cumulative_ah, 4),
                "Ah": round(ah, 4),
                "soc": round(soc, 4),
                "SoH_true": round(soh, 5),          # ground truth label (hidden from features)
                "Capacity_now": round(capacity_now, 5),
            })
    return pd.DataFrame(rows)


def generate_fleet(n_vehicles=5, n_cycles=120, out_path=None):
    dfs = [simulate_vehicle(v, n_cycles=n_cycles) for v in range(1, n_vehicles + 1)]
    fleet_df = pd.concat(dfs, ignore_index=True)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fleet_df.to_csv(out_path, index=False)
    return fleet_df


if __name__ == "__main__":
    df = generate_fleet(n_vehicles=5, n_cycles=120,
                         out_path=os.path.join(os.path.dirname(__file__), "fleet_raw.csv"))
    print(f"Generated {len(df)} rows across {df['Vehicle_ID'].nunique()} vehicles")
    print(df.head())
