"""Spatial merchant BTC and hydrogen investment components.

Uses standard PyPSA Generators, Links and Buses.
No imposed BTC/PEM capacity or hydrogen production targets.
Economic parameters must be supplied by the caller.

Hydrogen is sold locally as a product; no hydrogen network,
physical storage, or reconversion is introduced here.
"""

import numpy as np

BTC_CARRIER = "merchant bitcoin mining"
PEM_CARRIER = "merchant PEM electrolysis"
H2_CARRIER = "merchant H2 product"


def _nonnegative(value, name, positive=False):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} has an invalid value")
    return value


def add_merchant_consumers(n, electricity_buses, btc=None, h2=None):
    """Add endogenous BTC and/or PEM investment options at given AC buses.

    Parameters
    ----------
    btc : dict or None
        capital_cost_eur_per_mw_year
        revenue_eur_per_mwh_el
    h2 : dict or None
        capital_cost_eur_per_mw_year
        variable_cost_eur_per_mwh_el
        efficiency_lhv
        h2_lhv_kwh_per_kg
        h2_price_eur_per_kg
    """
    buses = list(electricity_buses)

    if not buses or len(buses) != len(set(buses)):
        raise ValueError("Bus list must be nonempty and unique")

    missing = set(buses) - set(n.buses.index)
    if missing:
        raise ValueError(f"Unknown buses: {sorted(missing)}")

    if any(n.buses.at[b, "carrier"] != "AC" for b in buses):
        raise ValueError("All candidate buses must be AC buses")

    if btc is None and h2 is None:
        raise ValueError("At least one consumer must be enabled")

    if btc is not None:
        btc_capex = _nonnegative(
            btc["capital_cost_eur_per_mw_year"], "BTC CAPEX"
        )
        btc_revenue = _nonnegative(
            btc["revenue_eur_per_mwh_el"], "BTC revenue"
        )

        if BTC_CARRIER not in n.carriers.index:
            n.add("Carrier", BTC_CARRIER)

    if h2 is not None:
        pem_capex = _nonnegative(
            h2["capital_cost_eur_per_mw_year"], "PEM CAPEX"
        )
        pem_vom = _nonnegative(
            h2["variable_cost_eur_per_mwh_el"], "PEM VOM"
        )
        eta = _nonnegative(
            h2["efficiency_lhv"], "PEM efficiency", positive=True
        )
        if eta > 1:
            raise ValueError("PEM efficiency cannot exceed 1")

        lhv = _nonnegative(
            h2["h2_lhv_kwh_per_kg"], "H2 LHV", positive=True
        )
        price = _nonnegative(
            h2["h2_price_eur_per_kg"], "H2 price"
        )

        for carrier in (PEM_CARRIER, H2_CARRIER):
            if carrier not in n.carriers.index:
                n.add("Carrier", carrier)

    for bus in buses:
        if btc is not None:
            name = f"{bus} merchant BTC"

            if name in n.generators.index:
                raise ValueError(f"Duplicate component: {name}")

            n.add(
                "Generator", name,
                bus=bus,
                carrier=BTC_CARRIER,
                sign=-1,
                p_nom_extendable=True,
                p_nom_min=0.0,
                p_min_pu=0.0,
                p_max_pu=1.0,
                capital_cost=btc_capex,
                marginal_cost=-btc_revenue,
            )

        if h2 is not None:
            h2_bus = f"{bus} merchant H2 bus"
            pem_name = f"{bus} merchant PEM"
            sale_name = f"{bus} merchant H2 sale"

            if (
                h2_bus in n.buses.index
                or pem_name in n.links.index
                or sale_name in n.generators.index
            ):
                raise ValueError(f"Duplicate H2 components at {bus}")

            metadata = {
                attr: n.buses.at[bus, attr]
                for attr in ("location", "x", "y")
                if attr in n.buses.columns
            }

            n.add(
                "Bus", h2_bus,
                carrier=H2_CARRIER,
                **metadata,
            )

            n.add(
                "Link", pem_name,
                bus0=bus,
                bus1=h2_bus,
                carrier=PEM_CARRIER,
                efficiency=eta,
                p_nom_extendable=True,
                p_nom_min=0.0,
                p_min_pu=0.0,
                p_max_pu=1.0,
                capital_cost=pem_capex,
                marginal_cost=pem_vom,
            )

            n.add(
                "Generator", sale_name,
                bus=h2_bus,
                carrier=H2_CARRIER,
                sign=-1,
                p_nom_extendable=True,
                p_nom_min=0.0,
                p_min_pu=0.0,
                p_max_pu=1.0,
                capital_cost=0.0,
                marginal_cost=-price * 1000.0 / lhv,
            )

    return {
        "candidate_buses": buses,
        "btc_enabled": btc is not None,
        "h2_enabled": h2 is not None,
    }
