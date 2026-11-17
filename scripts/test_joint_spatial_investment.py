"""Joint endogenous BTC and H2 investment: isolated regression test.

All prices, capacities and costs are synthetic test values.
No exogenous BTC/PEM capacity caps or H2 production target.
"""

import numpy as np
import pandas as pd
import pypsa

SITES = {
    "A": {"supply_mw": 10.0, "electricity_cost": 20.0, "h2_price": 3.0},
    "B": {"supply_mw": 8.0, "electricity_cost": 40.0, "h2_price": 2.5},
    "C": {"supply_mw": 6.0, "electricity_cost": 60.0, "h2_price": 3.0},
}

ETA = 0.64
LHV = 33.33
BTC_REVENUE = 50.0
BTC_CAPEX = 8.0
PEM_CAPEX = 20.0


def build_network(btc_enabled, h2_enabled):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2045-01-01", periods=4, freq="h"))

    for carrier in ("AC", "BTC", "PEM", "H2", "H2 product"):
        n.add("Carrier", carrier)

    for site, data in SITES.items():
        n.add("Bus", site, carrier="AC")

        # Finite physical electricity supply, not a BTC/PEM cap.
        n.add(
            "Generator",
            f"{site} supply",
            bus=site,
            carrier="AC",
            p_nom=data["supply_mw"],
            marginal_cost=data["electricity_cost"],
        )

        if btc_enabled:
            n.add(
                "Generator",
                f"{site} BTC",
                bus=site,
                carrier="BTC",
                sign=-1,
                p_nom_extendable=True,
                p_nom_min=0.0,
                capital_cost=BTC_CAPEX,
                marginal_cost=-BTC_REVENUE,
            )

        if h2_enabled:
            h2_bus = f"{site} hydrogen"
            n.add("Bus", h2_bus, carrier="H2")

            n.add(
                "Link",
                f"{site} PEM",
                bus0=site,
                bus1=h2_bus,
                carrier="PEM",
                efficiency=ETA,
                p_nom_extendable=True,
                p_nom_min=0.0,
                capital_cost=PEM_CAPEX,
            )

            n.add(
                "Generator",
                f"{site} H2 sale",
                bus=h2_bus,
                carrier="H2 product",
                sign=-1,
                p_nom_extendable=True,
                p_nom_min=0.0,
                capital_cost=0.0,
                marginal_cost=-data["h2_price"] * 1000.0 / LHV,
            )

    return n


def run_case(name, btc_enabled, h2_enabled):
    n = build_network(btc_enabled, h2_enabled)

    status, condition = n.optimize(solver_name="highs")

    assert (status, condition) == ("ok", "optimal"), (
        name, status, condition
    )

    print(f"\n===== {name} =====")

    for site, data in SITES.items():
        btc_capacity = (
            float(n.generators.at[f"{site} BTC", "p_nom_opt"])
            if btc_enabled else 0.0
        )
        pem_capacity = (
            float(n.links.at[f"{site} PEM", "p_nom_opt"])
            if h2_enabled else 0.0
        )

        btc_mwh = (
            float(n.generators_t.p[f"{site} BTC"].sum())
            if btc_enabled else 0.0
        )
        h2_mwh = (
            float(n.generators_t.p[f"{site} H2 sale"].sum())
            if h2_enabled else 0.0
        )
        pem_electricity = (
            float(n.links_t.p0[f"{site} PEM"].sum())
            if h2_enabled else 0.0
        )

        # The solution must respect physical electricity supply.
        assert btc_mwh + pem_electricity <= (
            4 * data["supply_mw"] + 1e-6
        )

        # Local H2 production and product delivery must balance.
        assert np.isclose(h2_mwh, ETA * pem_electricity, atol=1e-6)

        assert btc_capacity >= -1e-6
        assert pem_capacity >= -1e-6

        print(
            f"Site {site}: BTC={btc_capacity:.3f} MW, "
            f"PEM={pem_capacity:.3f} MW, "
            f"H2={h2_mwh:.3f} MWh"
        )

    print(f"Objective: {n.objective:.3f} EUR")
    return n


if __name__ == "__main__":
    btc = run_case("BTC ONLY", True, False)
    h2 = run_case("H2 ONLY", False, True)
    joint = run_case("JOINT BTC + H2", True, True)

    # In the joint case, the cheaper region A favours PEM
    # while region B favours BTC under the test assumptions.
    assert np.isclose(
        joint.links.at["A PEM", "p_nom_opt"], 10.0, atol=1e-6
    )
    assert np.isclose(
        joint.generators.at["B BTC", "p_nom_opt"], 8.0, atol=1e-6
    )

    assert np.isclose(
        joint.generators.at["A BTC", "p_nom_opt"], 0.0, atol=1e-6
    )
    assert np.isclose(
        joint.links.at["B PEM", "p_nom_opt"], 0.0, atol=1e-6
    )

    for name in ("C BTC",):
        assert np.isclose(
            joint.generators.at[name, "p_nom_opt"], 0.0, atol=1e-6
        )

    assert np.isclose(
        joint.links.at["C PEM", "p_nom_opt"], 0.0, atol=1e-6
    )

    print("\nALL JOINT SPATIAL INVESTMENT TESTS PASSED")
