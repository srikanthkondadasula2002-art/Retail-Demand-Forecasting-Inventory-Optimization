"""
Scenario Analysis & Price Elasticity Engine for Retail Demand Forecasting.
Simulates the impact of price drops (e.g., -10% promo) and pricing interventions
on sales demand trajectory, revenue, gross margin, and replenishment requirements.
"""

import math
import numpy as np
import pandas as pd

# Price Elasticity of Demand by Category
# E_d = % change in quantity / % change in price
CATEGORY_ELASTICITIES = {
    "Conservative": {
        "FOODS": -0.75,
        "HOUSEHOLD": -1.10,
        "HOBBIES": -1.50
    },
    "Moderate": {
        "FOODS": -0.95,
        "HOUSEHOLD": -1.35,
        "HOBBIES": -1.90
    },
    "Aggressive": {
        "FOODS": -1.20,
        "HOUSEHOLD": -1.65,
        "HOBBIES": -2.40
    }
}

# Cost of Goods Sold (COGS) benchmark as a fraction of base price
CATEGORY_COGS_RATIO = {
    "FOODS": 0.68,
    "HOUSEHOLD": 0.60,
    "HOBBIES": 0.52
}


def simulate_price_scenario(base_forecast_df: pd.DataFrame,
                            price_change_pct: float = -10.0,
                            elasticity_regime: str = "Moderate",
                            category_focus: str = "All Categories") -> dict:
    """
    Simulates demand and revenue trajectory under a price change scenario.
    
    Parameters:
        base_forecast_df: DataFrame with date, actual_units, lightgbm_forecast
        price_change_pct: Percentage change in price (e.g. -10.0 for 10% discount)
        elasticity_regime: 'Conservative', 'Moderate', or 'Aggressive'
        category_focus: 'All Categories', 'FOODS', 'HOUSEHOLD', or 'HOBBIES'
    
    Returns:
        dict containing:
            - trajectory_df: daily base vs simulated units & revenue
            - summary: high-level KPI impact metrics
            - category_breakdown: department/category impact analysis
    """
    pct_p = price_change_pct / 100.0
    elasticities = CATEGORY_ELASTICITIES.get(elasticity_regime, CATEGORY_ELASTICITIES["Moderate"])
    
    # Blended elasticity if all categories selected
    if category_focus == "All Categories":
        # M5 dataset approximate category sales weights: Foods 68.6%, Household 22.0%, Hobbies 9.4%
        blended_elasticity = (
            (elasticities["FOODS"] * 0.686) +
            (elasticities["HOUSEHOLD"] * 0.220) +
            (elasticities["HOBBIES"] * 0.094)
        )
        avg_base_price = 3.85
        avg_cogs_ratio = (0.68 * 0.686) + (0.60 * 0.220) + (0.52 * 0.094)
    else:
        blended_elasticity = elasticities.get(category_focus, -1.0)
        avg_base_price = 3.25 if category_focus == "FOODS" else (6.50 if category_focus == "HOUSEHOLD" else 8.20)
        avg_cogs_ratio = CATEGORY_COGS_RATIO.get(category_focus, 0.65)
        
    unit_cogs = avg_base_price * avg_cogs_ratio
    new_price = max(0.10, avg_base_price * (1.0 + pct_p))
    
    # Demand percentage change based on elasticity: %Delta Q = E_d * %Delta P
    # For a -10% price drop with E_d = -1.2, %Delta Q = -1.2 * -0.10 = +0.12 (+12%)
    pct_demand_lift = blended_elasticity * pct_p
    
    traj_rows = []
    total_base_units = 0
    total_sim_units = 0
    total_base_rev = 0.0
    total_sim_rev = 0.0
    total_base_margin = 0.0
    total_sim_margin = 0.0
    
    for _, row in base_forecast_df.iterrows():
        base_units = row["lightgbm_forecast"]
        # Add slight natural day-of-week promo elasticity boost on weekends
        d_obj = pd.to_datetime(row["date"])
        dow_promo_factor = 1.08 if d_obj.day_of_week in [5, 6] else 1.0
        
        sim_units = max(0, int(round(base_units * (1.0 + (pct_demand_lift * dow_promo_factor)))))
        
        base_rev = round(base_units * avg_base_price, 2)
        sim_rev = round(sim_units * new_price, 2)
        
        base_margin = round(base_units * (avg_base_price - unit_cogs), 2)
        sim_margin = round(sim_units * (new_price - unit_cogs), 2)
        
        total_base_units += base_units
        total_sim_units += sim_units
        total_base_rev += base_rev
        total_sim_rev += sim_rev
        total_base_margin += base_margin
        total_sim_margin += sim_margin
        
        traj_rows.append({
            "date": row["date"],
            "base_forecast": base_units,
            "simulated_demand": sim_units,
            "unit_lift": sim_units - base_units,
            "base_revenue": base_rev,
            "simulated_revenue": sim_rev,
            "base_margin": base_margin,
            "simulated_margin": sim_margin
        })
        
    df_traj = pd.DataFrame(traj_rows)
    
    # Calculate aggregate metrics
    unit_lift_total = total_sim_units - total_base_units
    pct_unit_lift_actual = ((total_sim_units - total_base_units) / max(1, total_base_units)) * 100.0
    rev_delta = total_sim_rev - total_base_rev
    pct_rev_delta = (rev_delta / max(1.0, total_base_rev)) * 100.0
    margin_delta = total_sim_margin - total_base_margin
    pct_margin_delta = (margin_delta / max(1.0, total_base_margin)) * 100.0
    
    # Additional buffer stock required to avoid stockouts during promotional surge
    extra_buffer_units_needed = int(math.ceil(unit_lift_total * 0.35))
    
    summary = {
        "price_change_pct": price_change_pct,
        "base_price": avg_base_price,
        "simulated_price": new_price,
        "total_base_units": total_base_units,
        "total_sim_units": total_sim_units,
        "unit_lift_total": unit_lift_total,
        "pct_unit_lift": pct_unit_lift_actual,
        "total_base_rev": total_base_rev,
        "total_sim_rev": total_sim_rev,
        "rev_delta": rev_delta,
        "pct_rev_delta": pct_rev_delta,
        "total_base_margin": total_base_margin,
        "total_sim_margin": total_sim_margin,
        "margin_delta": margin_delta,
        "pct_margin_delta": pct_margin_delta,
        "blended_elasticity": round(blended_elasticity, 2),
        "extra_buffer_units_needed": extra_buffer_units_needed
    }
    
    # Category Breakdown Matrix
    cat_rows = []
    for c, el in elasticities.items():
        c_price = 3.25 if c == "FOODS" else (6.50 if c == "HOUSEHOLD" else 8.20)
        c_new_price = c_price * (1.0 + pct_p)
        c_lift_pct = el * pct_p * 100.0
        c_cogs = c_price * CATEGORY_COGS_RATIO[c]
        c_base_margin_pct = ((c_price - c_cogs) / c_price) * 100.0
        c_new_margin_pct = ((c_new_price - c_cogs) / max(0.01, c_new_price)) * 100.0
        
        cat_rows.append({
            "Category": c,
            "Price Elasticity (Ed)": el,
            "Base Price ($)": f"${c_price:.2f}",
            "Promo Price ($)": f"${c_new_price:.2f}",
            "Projected Demand Lift (%)": f"{c_lift_pct:+.1f}%",
            "Base Margin (%)": f"{c_base_margin_pct:.1f}%",
            "Promo Margin (%)": f"{c_new_margin_pct:.1f}%",
            "Impact Type": "Volume Driver 🚀" if abs(el) > 1.4 else "Margin Stable ⚖️"
        })
        
    df_cat = pd.DataFrame(cat_rows)
    
    return {
        "trajectory_df": df_traj,
        "summary": summary,
        "category_breakdown": df_cat
    }
