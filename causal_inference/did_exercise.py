"""
Difference-in-differences: did the West Texas battery lower peak prices?

A self-directed practice exercise in causal inference, on ERCOT-shaped data.

Setup
-----
A large battery storage project comes online in the ERCOT West zone on
2025-01-01. The question is whether it reduced peak prices, and by how much.

data/ercot_did_peak_prices.csv has daily peak prices for four ERCOT zones
across 2024 and 2025:

    date            day
    zone            WEST, NORTH, HOUSTON, SOUTH
    peak_price_mwh  $/MWh, the outcome
    peak_load_mw    zonal load at the peak hour

WEST is the treated zone. NORTH, HOUSTON, and SOUTH are untreated. There is
a confounder that is not a column in the CSV and that moves all four zones
at once, which is what makes the naive pre/post comparison misleading.

The data is synthetic. It was generated using Claude with a known treatment
effect so the estimates below could be checked against ground truth. It is
not real ERCOT market data and should not be used as such.

Approach
--------
1. The naive pre/post change in WEST alone, to see what it misses.
2. DiD by hand, as a 2x2 table of group means.
3. The same estimate as an OLS regression with a treated:post interaction,
   which also gives standard errors.
4. A parallel trends check, since the whole design rests on that assumption.

Run with: .venv/bin/python causal_inference/did_exercise.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

DATA = "data/ercot_did_peak_prices.csv"
TREATMENT_START = pd.Timestamp("2025-01-01")
TREATED_ZONE = "WEST"


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"])
    df['treated'] = (df.zone == TREATED_ZONE).astype(int) # binary indicating if the zone is treated
    df['post'] = (df.date >= TREATMENT_START).astype(int) # binary indicating if the date is after treatment
    return df


def naive_pre_post(df: pd.DataFrame) -> float:
    """Pre/post change in WEST only.

    What this number is actually measuring:
    this number is measuring the difference in the average peak price in WEST zone before and after the asset came online.
    In this case, the seasonality is equally distributed over the pre and post period. However, other factors can influence the result here.
    For example, ERCOT saw generally lower distribution of prices in 2025 relative to 2024, which would bias the results to look like the battery
    had a larger impact than it actually did. This number is the battery effect plus everything else that changed in ERCOT between 2024 and 2025.
    """
    pre_change =df[df["zone"] == TREATED_ZONE].groupby(df.date < TREATMENT_START).peak_price_mwh.mean()[True]
    post_change = df[df["zone"] == TREATED_ZONE].groupby(df.date >= TREATMENT_START).peak_price_mwh.mean()[True]
    return post_change - pre_change


def did_by_hand(df: pd.DataFrame) -> float:
    """The 2x2 table and (b - a) - (d - c).

    How I aggregated the control zones, and why:
    I will aggregate all non-WEST zones together, because the battery is in West, so it acts as the test group, whereas the other zones act as the control group
    """
    # a = pre-treatment for WEST; b = post-treatment for WEST; c = pre-treatment for controls; d = post-treatment for controls
    a = df[(df["zone"] == TREATED_ZONE) & (df["date"] < TREATMENT_START)].peak_price_mwh.mean()
    b = df[(df["zone"] == TREATED_ZONE) & (df["date"] >= TREATMENT_START)].peak_price_mwh.mean()
    c = df[(df["zone"] != TREATED_ZONE) & (df["date"] < TREATMENT_START)].peak_price_mwh.mean()
    d = df[(df["zone"] != TREATED_ZONE) & (df["date"] >= TREATMENT_START)].peak_price_mwh.mean()
    did = (b-a) - (d-c)
    return did


def did_regression(df: pd.DataFrame):
    """OLS with the treated:post interaction.

    Whether the interaction coefficient matches the hand calculation:
    Yes! the coef for treated:post is the same as the step 2 estimate, -8.4.

    Whether the standard error is trustworthy:
    No, the observations are correlated. For example, West's prices on Jan 2nd will be related to its prices on Jan 1st, even if they are defined as different regimes here.
    Similarly, West's prices on any given date will be related to the other zones prices, because they are all made up of the MEC + MCC, and MEC is the same across all of ERCOT.
    """
    # ordinary least squares regression 
    model_result = smf.ols("peak_price_mwh ~ treated + post + treated:post", data=df).fit()
    return model_result.summary()


def parallel_trends_check(df: pd.DataFrame) -> None:
    """Monthly means, WEST vs. controls, line at treatment.

    Whether the pre-period series move together:
    Yes they do move together, especially this is evident in how they both peak in Aug 2024 when temperatures were likely high. Also their difference is consistent month to month pre-battery coming online.
    However, a noticeable shift happens after the battery comes online: the test - control drops dramatically, suggesting that the battery did have a meaningful impact on prices.

    What I would tell a stakeholder who asks how I know the control zones are a
    fair comparison:
    Three parts.

    1. I cannot prove it. The counterfactual is unobservable by construction, so
       parallel trends is an assumption, not a finding. What I can show is that it
       held for every month I am able to check. The gap between West and the control
       zones was stable across all of 2024, which means the ways West differs from
       Houston show up as a constant offset rather than a diverging trend. A constant
       offset is exactly what DiD is built to remove.

    2. The control zones are defensible on substance, not just on the plot. All four
       zones sit in the same market under the same operator, clear against the same
       supply stack, and share the energy component of the LMP. When prices move,
       they move together. That shared structure is why the control zones can stand
       in for the ERCOT-wide conditions West was also exposed to.

    3. The real risk is not that the control zones are wrong, it is that something
       else hit West alone in January 2025. The estimate credits the battery with the
       entire change in the gap, so any West-specific event with similar timing is
       indistinguishable from it. Candidates worth ruling out:

         - New wind or solar capacity in West. West holds most of ERCOT's renewable
           buildout, and new wind in particular suppresses evening peak prices.
           Check ERCOT capacity by zone for a step change in early 2025.
         - Transmission upgrades relieving congestion into or out of West. This pulls
           West prices toward the rest of the market and produces the same downward
           step in the gap.
         - Load growth in the Permian from oil and gas electrification or data
           centers. This pushes the other way, so if it is present my -8.41 is an
           understatement.
         - Other storage. If several batteries came online in West at once, the
           estimate is the effect of the storage buildout, not of one project.

    Two checks I would run. First, decompose the LMP into its energy and congestion
    components. If the drop sits in the congestion component rather than the energy
    component, that points at transmission, not at the battery. Second, a placebo
    test: rerun the same DiD with a fake treatment date in mid-2024, when nothing
    happened. It should return an estimate near zero. If it does not, the design is
    picking up something other than the battery.
    """
    df["month"] = df.date.dt.to_period("M").dt.to_timestamp()
    grouped_monthly = df.groupby(["month", "treated"])["peak_price_mwh"].mean()

    # one column per group: 0 = controls, 1 = WEST
    monthly = grouped_monthly.unstack("treated").rename(columns={0: "control", 1: "test"})
    monthly["gap"] = monthly["test"] - monthly["control"]

    pre_gap = monthly.loc[monthly.index < TREATMENT_START, "gap"].mean()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- top: the two series -------------------------------------------------
    ax1.plot(monthly.index, monthly["control"], "o-", color="#4C78A8",
             label="control (NORTH/HOUSTON/SOUTH)")
    ax1.plot(monthly.index, monthly["test"], "o-", color="#E45756",
             label=f"test ({TREATED_ZONE})")
    ax1.set_ylabel("mean peak price ($/MWh)")
    ax1.set_title("Parallel trends check: monthly mean peak price")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    # --- bottom: the gap between them ---------------------------------------
    ax2.plot(monthly.index, monthly["gap"], "o-", color="#54A24B")
    ax2.axhline(pre_gap, color="gray", ls=":", lw=1.5,
                label=f"pre-period mean gap = {pre_gap:.2f}")
    ax2.set_ylabel("test - control ($/MWh)")
    ax2.set_xlabel("month")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.25)

    for ax in (ax1, ax2):
        ax.axvline(TREATMENT_START, color="black", ls="--", lw=1.5)
    ax1.annotate("battery online", xy=(TREATMENT_START, ax1.get_ylim()[1]),
                 xytext=(6, -14), textcoords="offset points", fontsize=9)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load()
    print(f"{len(df)} rows, {df.date.min().date()} to {df.date.max().date()}")
    print(df.zone.value_counts().to_string())

    print("\nnaive pre/post:      ", naive_pre_post(df))
    print("DiD by hand:         ", did_by_hand(df))
    print("DiD regression:")
    print(did_regression(df))
    parallel_trends_check(df)
