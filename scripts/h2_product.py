"""Hydrogen product modelling using standard PyPSA components.

Hydrogen is sold as a product, not stored for reconversion.
Supports fixed annual demand and endogenous merchant production.
Compatible with PyPSA 0.30.x.
"""

import numpy as np
import xarray as xr

PEM_NAME = "H2 product PEM electrolyser"
H2_BUS = "H2 product bus"
OFFTAKE_NAME = "H2 product offtake"
CONSTRAINT_NAME = "H2-annual-product-requirement"


def add_h2_product(n, connection_bus, parameters, mode):
    """Attach a PEM Link and a negative-sign product offtake Generator.

    All energy flows use MWh on an LHV basis.
    Capital costs are expressed in EUR/MW/year.
    Variable electricity-input costs are expressed in EUR/MWh_el.
    """

    if mode not in ("fixed", "merchant"):
        raise ValueError("H2 mode must be 'fixed' or 'merchant'.")

    if connection_bus not in n.buses.index:
        raise ValueError(f"Unknown connection bus: {connection_bus}")

    if PEM_NAME in n.links.index or OFFTAKE_NAME in n.generators.index:
        raise ValueError("H2 product components already exist.")

    efficiency = float(parameters["efficiency_lhv"])
    capacity = float(parameters["p_nom_mw"])
    capital_cost = float(parameters["capital_cost_eur_per_mw_year"])
    variable_cost = float(parameters["variable_cost_eur_per_mwh_el"])
    lhv = float(parameters["h2_lhv_kwh_per_kg"])

    values = [efficiency, capacity, capital_cost, variable_cost, lhv]
    if not all(np.isfinite(v) for v in values):
        raise ValueError("All H2 parameters must be finite.")

    if not 0 < efficiency <= 1:
        raise ValueError("Invalid PEM efficiency.")
    if capacity <= 0 or capital_cost < 0 or variable_cost < 0 or lhv <= 0:
        raise ValueError("Invalid PEM technical or economic parameters.")

    if mode == "merchant":
        capacity_max = float(parameters["p_nom_max_mw"])
        price = float(parameters["h2_price_eur_per_kg"])
        if not np.isfinite(capacity_max) or capacity_max <= 0:
            raise ValueError("Merchant capacity limit must be positive.")
        if not np.isfinite(price) or price < 0:
            raise ValueError("Merchant H2 price must be nonnegative.")
    else:
        capacity_max = capacity
        price = 0.0
        target_kt = float(parameters["annual_h2_kt"])
        if not np.isfinite(target_kt) or target_kt <= 0:
            raise ValueError("Fixed H2 target must be positive.")

        target_mwh_h2 = target_kt * 1_000_000 * lhv / 1000

        available_hours = float(
            n.snapshot_weightings.generators.sum()
        )
        maximum_h2 = capacity * efficiency * available_hours

        if target_mwh_h2 > maximum_h2 + 1e-5:
            raise ValueError("Annual H2 target exceeds PEM capability.")

    for carrier in ("H2 product", "H2 product PEM"):
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)

    if H2_BUS in n.buses.index:
        raise ValueError("H2 product bus already exists.")

    # Inherit available geographical metadata from the electricity bus.
    # Minimal PyPSA networks do not necessarily contain these columns.
    bus_metadata = {}

    for attribute in ("location", "x", "y"):
        if attribute in n.buses.columns:
            bus_metadata[attribute] = n.buses.at[
                connection_bus, attribute
            ]

    n.add(
        "Bus",
        H2_BUS,
        carrier="H2 product",
        **bus_metadata,
    )

    n.add(
        "Link",
        PEM_NAME,
        bus0=connection_bus,
        bus1=H2_BUS,
        carrier="H2 product PEM",
        efficiency=efficiency,
        p_nom=capacity if mode == "fixed" else 0.0,
        p_nom_extendable=(mode == "merchant"),
        p_nom_min=0.0,
        p_nom_max=capacity_max if mode == "merchant" else np.inf,
        p_min_pu=0.0,
        p_max_pu=1.0,
        capital_cost=capital_cost,
        marginal_cost=variable_cost,
    )

    # Product demand in MW_H2. This is NOT an H2 generator.
    # The negative sign makes positive dispatch consume H2.
    n.add(
        "Generator",
        OFFTAKE_NAME,
        bus=H2_BUS,
        carrier="H2 product",
        sign=-1,
        p_nom=capacity_max * efficiency,
        p_nom_extendable=False,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=-price * 1000 / lhv,
    )

    if mode == "fixed":
        return target_mwh_h2

    return None


def add_fixed_h2_requirement(n, snapshots, target_mwh_h2):
    """Enforce annual product delivery, without hourly demand targets."""

    if target_mwh_h2 is None:
        return

    production = n.model["Generator-p"].sel(
        Generator=OFFTAKE_NAME
    )

    weights = xr.DataArray(
        n.snapshot_weightings.generators.loc[snapshots].to_numpy(),
        coords={"snapshot": snapshots},
        dims=["snapshot"],
    )

    n.model.add_constraints(
        (production * weights).sum("snapshot") == target_mwh_h2,
        name=CONSTRAINT_NAME,
    )
