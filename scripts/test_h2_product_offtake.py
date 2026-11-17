"""Test hydrogen product offtake using standard PyPSA components."""

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

EFFICIENCY = 0.64
H2_LHV_KWH_PER_KG = 33.33
TARGET_MWH_H2 = 12.8


def build_network(mode, price_eur_per_kg=0.0):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2045-01-01", periods=4, freq="h"))

    for carrier in ["AC", "H2", "H2 electrolysis", "H2 product"]:
        n.add("Carrier", carrier)

    n.add("Bus", "electricity", carrier="AC")
    n.add("Bus", "hydrogen-product", carrier="H2")

    n.add(
        "Generator",
        "electricity-supply",
        bus="electricity",
        carrier="AC",
        p_nom=100.0,
        marginal_cost=40.0,
    )

    n.add(
        "Link",
        "PEM-electrolysis",
        bus0="electricity",
        bus1="hydrogen-product",
        carrier="H2 electrolysis",
        efficiency=EFFICIENCY,
        p_nom=10.0 if mode == "fixed" else 0.0,
        p_nom_extendable=(mode == "market"),
        p_nom_max=10.0,
        capital_cost=20.0 if mode == "market" else 0.0,
        marginal_cost=0.0,
    )

    # Revenue is measured per MWh of hydrogen product (LHV).
    revenue_eur_per_mwh_h2 = (
        price_eur_per_kg * 1000.0 / H2_LHV_KWH_PER_KG
    )

    n.add(
        "Generator",
        "H2-product-offtake",
        bus="hydrogen-product",
        carrier="H2 product",
        sign=-1,
        p_nom=100.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=-revenue_eur_per_mwh_h2,
    )

    return n


def fixed_production_constraint(n, snapshots):
    """Enforce exact annual H2 product delivery."""
    product = n.model["Generator-p"].sel(
        Generator="H2-product-offtake"
    )

    weights = xr.DataArray(
        n.snapshot_weightings.generators.loc[snapshots].to_numpy(),
        coords={"snapshot": snapshots},
        dims=["snapshot"],
    )

    n.model.add_constraints(
        (product * weights).sum("snapshot") == TARGET_MWH_H2,
        name="H2-annual-product-requirement",
    )


def run_case(label, mode, price=0.0):
    n = build_network(mode, price)

    kwargs = {"solver_name": "highs"}
    if mode == "fixed":
        kwargs["extra_functionality"] = fixed_production_constraint

    status, condition = n.optimize(**kwargs)

    assert status == "ok" and condition == "optimal"

    weights = n.snapshot_weightings.generators

    electricity = float(
        (n.links_t.p0["PEM-electrolysis"] * weights).sum()
    )
    hydrogen = float(
        (n.generators_t.p["H2-product-offtake"] * weights).sum()
    )

    assert np.isclose(
        hydrogen, electricity * EFFICIENCY, atol=1e-6
    )

    capacity = float(n.links.at["PEM-electrolysis", "p_nom_opt"])

    print(
        f"{label}: PEM={capacity:.3f} MW, "
        f"electricity={electricity:.3f} MWh, "
        f"H2={hydrogen:.3f} MWh_H2"
    )

    return capacity, hydrogen


if __name__ == "__main__":
    _, fixed_h2 = run_case("FIXED", "fixed")
    low_capacity, low_h2 = run_case("LOW PRICE", "market", 1.0)
    high_capacity, high_h2 = run_case("HIGH PRICE", "market", 3.0)

    assert np.isclose(fixed_h2, TARGET_MWH_H2, atol=1e-6)
    assert np.isclose(low_capacity, 0.0, atol=1e-6)
    assert np.isclose(low_h2, 0.0, atol=1e-6)
    assert np.isclose(high_capacity, 10.0, atol=1e-6)
    assert np.isclose(high_h2, 25.6, atol=1e-6)

    print("ALL H2 PRODUCT OFFTAKE TESTS PASSED")
