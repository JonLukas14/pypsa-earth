"""Regression tests for the reusable H2 product module."""

import numpy as np
import pandas as pd
import pypsa

from h2_product import (
    PEM_NAME,
    OFFTAKE_NAME,
    add_h2_product,
    add_fixed_h2_requirement,
)


def make_network():
    n = pypsa.Network()
    n.set_snapshots(
        pd.date_range("2045-01-01", periods=4, freq="h")
    )

    n.add("Carrier", "AC")
    n.add("Bus", "electricity", carrier="AC")

    n.add(
        "Generator",
        "electricity supply",
        bus="electricity",
        carrier="AC",
        p_nom=100.0,
        marginal_cost=40.0,
    )
    return n


def parameters(price=0.0):
    return {
        "efficiency_lhv": 0.64,
        "p_nom_mw": 10.0,
        "p_nom_max_mw": 10.0,
        "capital_cost_eur_per_mw_year": 20.0,
        "variable_cost_eur_per_mwh_el": 0.0,
        "h2_lhv_kwh_per_kg": 33.33,
        "h2_price_eur_per_kg": price,
        "annual_h2_kt": 12.8 / 33330.0,
    }


def test_case(mode, price=0.0):
    n = make_network()

    target = add_h2_product(
        n=n,
        connection_bus="electricity",
        parameters=parameters(price),
        mode=mode,
    )

    def extra_functionality(network, snapshots):
        add_fixed_h2_requirement(
            network, snapshots, target
        )

    kwargs = {"solver_name": "highs"}

    if mode == "fixed":
        kwargs["extra_functionality"] = extra_functionality

    status, condition = n.optimize(**kwargs)

    assert status == "ok", (status, condition)
    assert condition == "optimal", (status, condition)

    weights = n.snapshot_weightings.generators

    electricity = float(
        (n.links_t.p0[PEM_NAME] * weights).sum()
    )

    hydrogen = float(
        (n.generators_t.p[OFFTAKE_NAME] * weights).sum()
    )

    capacity = float(
        n.links.at[PEM_NAME, "p_nom_opt"]
    )

    assert np.isclose(
        hydrogen, electricity * 0.64, atol=1e-6
    ), "Hydrogen energy balance failed"

    if mode == "fixed":
        assert np.isclose(hydrogen, 12.8, atol=1e-6)
    elif price == 1.0:
        assert np.isclose(capacity, 0.0, atol=1e-6)
        assert np.isclose(hydrogen, 0.0, atol=1e-6)
    elif price == 3.0:
        assert np.isclose(capacity, 10.0, atol=1e-6)
        assert np.isclose(hydrogen, 25.6, atol=1e-6)

    print(
        f"{mode.upper()}, price={price:.1f}: "
        f"PEM={capacity:.3f} MW, "
        f"electricity={electricity:.3f} MWh, "
        f"H2={hydrogen:.3f} MWh, "
        f"objective={n.objective:.3f} EUR"
    )


if __name__ == "__main__":
    test_case("fixed")
    test_case("merchant", 1.0)
    test_case("merchant", 3.0)
    print("ALL H2 MODULE TESTS PASSED")
