"""Isolated PyPSA 0.30.x hydrogen-product architecture test."""

import numpy as np
import pandas as pd
import pypsa


EFFICIENCY = 0.64
H2_LHV_KWH_PER_KG = 33.33


def build_test_network(mode, h2_price_eur_per_kg=0.0):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2045-01-01", periods=4, freq="h"))

    n.snapshot_weightings.loc[:, :] = 1.0

    n.add("Bus", "electricity", carrier="AC")
    n.add("Bus", "hydrogen-product", carrier="H2")

    # Toy electricity supply: 40 EUR/MWh.
    n.add(
        "Generator",
        "electricity-supply",
        bus="electricity",
        p_nom=100.0,
        marginal_cost=40.0,
    )

    fixed_mode = mode == "fixed"

    # Revenue per MWh of electricity entering the PEM Link.
    revenue_per_mwh_el = (
        h2_price_eur_per_kg
        * EFFICIENCY
        * 1000.0
        / H2_LHV_KWH_PER_KG
    )

    n.add(
        "Link",
        "PEM-electrolysis",
        bus0="electricity",
        bus1="hydrogen-product",
        carrier="H2 electrolysis",
        efficiency=EFFICIENCY,
        p_nom=10.0 if fixed_mode else 0.0,
        p_nom_extendable=not fixed_mode,
        p_nom_max=10.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        capital_cost=0.0 if fixed_mode else 20.0,
        marginal_cost=0.0 if fixed_mode else -revenue_per_mwh_el,
    )

    # Product-accounting Store, not a physical H2 storage plant.
    # The H2 bus has no reconversion link.
    target_mwh_h2 = 12.8
    e_nom = target_mwh_h2 if fixed_mode else 30.0

    e_min_pu = pd.Series(0.0, index=n.snapshots)
    e_max_pu = pd.Series(1.0, index=n.snapshots)

    if fixed_mode:
        e_min_pu.iloc[-1] = 1.0

    n.add(
        "Store",
        "hydrogen-product-account",
        bus="hydrogen-product",
        e_nom=e_nom,
        e_nom_extendable=False,
        e_initial=0.0,
        e_cyclic=False,
        e_min_pu=e_min_pu,
        e_max_pu=e_max_pu,
        standing_loss=0.0,
        capital_cost=0.0,
        marginal_cost=0.0,
    )

    return n


def run_case(name, mode, price=0.0):
    n = build_test_network(mode, price)
    status, condition = n.optimize(solver_name="highs")

    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"{name}: optimization failed: {status}, {condition}"
        )

    electricity_mwh = float(
        (n.links_t.p0["PEM-electrolysis"]
         * n.snapshot_weightings["generators"]).sum()
    )
    hydrogen_mwh = float(
        n.stores_t.e["hydrogen-product-account"].iloc[-1]
    )
    hydrogen_kg = hydrogen_mwh * 1000 / H2_LHV_KWH_PER_KG

    expected_hydrogen = EFFICIENCY * electricity_mwh
    assert np.isclose(hydrogen_mwh, expected_hydrogen, atol=1e-5)

    capacity_mw = float(n.links.at["PEM-electrolysis", "p_nom_opt"])

    print(
        f"{name}: PEM={capacity_mw:.3f} MW, "
        f"electricity={electricity_mwh:.3f} MWh, "
        f"H2={hydrogen_mwh:.3f} MWh_H2 "
        f"({hydrogen_kg:.2f} kg)"
    )

    return capacity_mw, hydrogen_mwh


if __name__ == "__main__":
    _, fixed_h2 = run_case("FIXED", "fixed")
    low_capacity, low_h2 = run_case("LOW PRICE", "market", 1.0)
    high_capacity, high_h2 = run_case("HIGH PRICE", "market", 3.0)

    assert np.isclose(fixed_h2, 12.8, atol=1e-5)
    assert np.isclose(low_capacity, 0.0, atol=1e-5)
    assert np.isclose(low_h2, 0.0, atol=1e-5)
    assert np.isclose(high_capacity, 10.0, atol=1e-5)
    assert np.isclose(high_h2, 25.6, atol=1e-5)

    print("ALL H2 PRODUCT ARCHITECTURE TESTS PASSED")
