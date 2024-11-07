"""
Inventory Optimization Engine for Retail Demand Forecasting Dashboard.
Provides safety stock, reorder point (ROP), economic order quantity (EOQ),
and replenishment recommendations based on lead-time demand variability.
"""

import math
import numpy as np
import pandas as pd

# Standard Normal Distribution Z-Scores for Common Service Levels
SERVICE_LEVEL_Z = {
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326
}

# Category Lead Time (days) and Demand Variability Profiles
CATEGORY_PROFILES = {
    "FOODS": {
        "lead_time_days": 4,
        "lead_time_std": 1.0,
        "holding_cost_rate": 0.25,
        "order_cost": 30.0,
        "demand_cv": 0.35  # coefficient of variation (sigma / mu)
    },
    "HOUSEHOLD": {
        "lead_time_days": 8,
        "lead_time_std": 1.5,
        "holding_cost_rate": 0.20,
        "order_cost": 45.0,
        "demand_cv": 0.45
    },
    "HOBBIES": {
        "lead_time_days": 12,
        "lead_time_std": 2.0,
        "holding_cost_rate": 0.18,
        "order_cost": 50.0,
        "demand_cv": 0.60
    }
}


def calculate_safety_stock(mean_daily_demand: float,
                           std_daily_demand: float,
                           lead_time_days: float,
                           lead_time_std: float,
                           service_level: float = 0.95) -> int:
    """
    Calculates statistical safety stock accounting for both demand uncertainty
    and supplier lead-time uncertainty:
    SS = Z * sqrt(L * sigma_d^2 + d_avg^2 * sigma_L^2)
    """
    z = SERVICE_LEVEL_Z.get(round(service_level, 2), 1.645)
    variance_term = (lead_time_days * (std_daily_demand ** 2)) + ((mean_daily_demand ** 2) * (lead_time_std ** 2))
    safety_stock = z * math.sqrt(max(0, variance_term))
    return int(math.ceil(safety_stock))


def calculate_reorder_point(mean_daily_demand: float,
                            lead_time_days: float,
                            safety_stock: int) -> int:
    """
    ROP = (Mean Daily Demand * Lead Time) + Safety Stock
    """
    lead_time_demand = mean_daily_demand * lead_time_days
    return int(math.ceil(lead_time_demand + safety_stock))


def calculate_eoq(annual_demand: float,
                  order_cost: float,
                  holding_cost_unit: float) -> int:
    """
    Economic Order Quantity (Wilson formula):
    EOQ = sqrt( (2 * D * S) / H )
    """
    if holding_cost_unit <= 0 or annual_demand <= 0:
        return 50
    eoq = math.sqrt((2 * annual_demand * order_cost) / holding_cost_unit)
    return max(10, int(round(eoq)))


def evaluate_stock_status(current_stock: int, safety_stock: int, reorder_point: int, eoq: int) -> str:
    """Categorizes stock health status."""
    if current_stock < safety_stock:
        return "🔴 Critical Risk"
    elif current_stock <= reorder_point:
        return "🟡 Reorder Needed"
    elif current_stock <= (reorder_point + eoq * 1.5):
        return "🟢 Healthy"
    else:
        return "🔵 Overstocked"


def generate_inventory_recommendations(df_products: pd.DataFrame,
                                       store_id: str = "CA_1",
                                       service_level: float = 0.95,
                                       order_cost_override: float = None,
                                       holding_cost_pct_override: float = None) -> pd.DataFrame:
    """
    Simulates inventory levels and generates replenishment recommendations
    for catalog SKUs across stores.
    """
    np.random.seed(hash(store_id) % 100000 + 101)
    results = []

    store_multiplier = {
        "CA_1": 1.05, "CA_2": 0.98, "CA_3": 1.25, "CA_4": 0.85,
        "TX_1": 0.90, "TX_2": 0.95, "TX_3": 0.92,
        "WI_1": 0.88, "WI_2": 1.10, "WI_3": 0.87
    }.get(store_id, 1.0)

    for _, row in df_products.iterrows():
        cat = row["category"]
        profile = CATEGORY_PROFILES.get(cat, CATEGORY_PROFILES["FOODS"])

        lead_time = profile["lead_time_days"]
        lead_time_std = profile["lead_time_std"]
        order_cost = order_cost_override if order_cost_override else profile["order_cost"]
        holding_pct = (holding_cost_pct_override / 100.0) if holding_cost_pct_override else profile["holding_cost_rate"]

        price = row["base_price"]
        holding_cost_unit = max(0.20, price * holding_pct)

        # Base daily demand calibrated by store scale
        avg_daily = max(1.0, (row["sales_velocity"] / 30.0) * store_multiplier * np.random.uniform(0.9, 1.1))
        std_daily = avg_daily * profile["demand_cv"]
        annual_demand = avg_daily * 365

        # Calculations
        ss = calculate_safety_stock(avg_daily, std_daily, lead_time, lead_time_std, service_level)
        rop = calculate_reorder_point(avg_daily, lead_time, ss)
        eoq = calculate_eoq(annual_demand, order_cost, holding_cost_unit)

        # Simulated on-hand stock: realistic mix of low, medium, and high stock
        stock_scenario = np.random.choice(["critical", "reorder", "healthy", "healthy_high", "excess"],
                                          p=[0.15, 0.25, 0.40, 0.15, 0.05])
        if stock_scenario == "critical":
            current_stock = int(np.random.uniform(0.2, 0.9) * ss)
        elif stock_scenario == "reorder":
            current_stock = int(np.random.uniform(ss, rop))
        elif stock_scenario == "healthy":
            current_stock = int(np.random.uniform(rop, rop + (eoq * 0.7)))
        elif stock_scenario == "healthy_high":
            current_stock = int(np.random.uniform(rop + (eoq * 0.7), rop + (eoq * 1.3)))
        else:
            current_stock = int(rop + (eoq * 1.8) + np.random.uniform(20, 100))

        status = evaluate_stock_status(current_stock, ss, rop, eoq)
        days_of_supply = round(current_stock / avg_daily, 1)

        # Replenishment Order Logic
        if current_stock <= rop:
            reorder_units = max(eoq, (rop + eoq) - current_stock)
            urgency = "HIGH" if status == "🔴 Critical Risk" else "MEDIUM"
        else:
            reorder_units = 0
            urgency = "NONE"

        est_order_cost = round(reorder_units * price, 2)
        stock_value = round(current_stock * price, 2)

        results.append({
            "store": store_id,
            "item_id": row["item_id"],
            "item_name": row["item_name"],
            "category": cat,
            "department": row["department"],
            "unit_price": price,
            "daily_demand": round(avg_daily, 1),
            "lead_time_days": lead_time,
            "safety_stock": ss,
            "reorder_point": rop,
            "eoq": eoq,
            "current_stock": current_stock,
            "days_of_supply": days_of_supply,
            "stock_status": status,
            "reorder_units": reorder_units,
            "order_urgency": urgency,
            "est_order_cost": est_order_cost,
            "stock_value": stock_value
        })

    return pd.DataFrame(results)
