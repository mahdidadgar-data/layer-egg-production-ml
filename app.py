"""Layer flock planner: Streamlit app.

Run:  streamlit run app.py
"""
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FLOCKS_FILE, REPORTS_DIR, WEEKLY_FILE
from src.diet_builder import build_diets, diets_table, table_to_diets
from src.modeling import split_farms
from src.predict import get_predictor

st.set_page_config(page_title="Layer flock planner", page_icon="🥚", layout="wide")

TARGET_LABELS = {
    "hen_day_pct": "Hen-day production (%)",
    "egg_weight_g": "Egg weight (g)",
    "breaking_strength_n": "Shell breaking strength (N)",
    "haugh_units": "Haugh units",
}


@st.cache_resource
def predictor():
    return get_predictor()


@st.cache_data
def load_records():
    flocks = pd.read_csv(FLOCKS_FILE)
    weekly = pd.read_csv(WEEKLY_FILE)
    _, test_farms = split_farms(flocks)
    examples = flocks[flocks.farm_id.isin(test_farms) & (flocks.last_recorded_age_wk >= 50)]
    return flocks, weekly, examples


@st.cache_data
def load_report(name):
    return pd.read_csv(REPORTS_DIR / name)


def age_axis(lo, hi):
    return alt.X("age_wk:Q", title="Age (weeks)", scale=alt.Scale(domain=[int(lo), int(hi)], nice=False))


def band_chart(df, y, lower, upper, title, standard=None, y_domain=None):
    base = alt.Chart(df).encode(x=age_axis(df.age_wk.min(), df.age_wk.max()))
    scale = alt.Scale(domain=y_domain, zero=False) if y_domain else alt.Scale(zero=False)
    band = base.mark_area(opacity=0.2, color="#2E7D32").encode(
        y=alt.Y(f"{lower}:Q", title=title, scale=scale), y2=f"{upper}:Q")
    line = base.mark_line(color="#2E7D32", strokeWidth=2.5).encode(
        y=f"{y}:Q", tooltip=[alt.Tooltip("age_wk:Q", title="Age"), alt.Tooltip(f"{y}:Q", title="Predicted", format=".1f")])
    layers = [band, line]
    if standard:
        layers.append(base.mark_line(color="#757575", strokeDash=[5, 4]).encode(
            y=f"{standard}:Q", tooltip=[alt.Tooltip(f"{standard}:Q", title="Breed standard", format=".1f")]))
    return alt.layer(*layers).properties(height=300)


# ---------------- Sidebar: the flock plan ----------------
st.sidebar.header("Flock plan")
strain = st.sidebar.selectbox("Strain", ["Lohmann Brown-Classic", "Hy-Line W-80"])
housing = st.sidebar.selectbox("Housing", ["enriched cage", "barn", "free range"], index=1)
zone = st.sidebar.selectbox("Climate", ["temperate", "warm"])
start = st.sidebar.date_input("Date the flock reaches 19 weeks", date(2026, 3, 2))
pullet_bw = st.sidebar.slider("Pullet weight at transfer (% of standard)", 88, 112, 100)
photostim = st.sidebar.select_slider("Age at light stimulation (weeks)", [16, 17, 18, 19], 17)
end_age = st.sidebar.slider("End of lay (weeks)", 72, 90, 80)
c1, c2 = st.sidebar.columns(2)
p2 = c1.number_input("Phase 2 from week", 46, 56, 50)
p3 = c2.number_input("Phase 3 from week", 60, 76, 70)

st.sidebar.subheader("Diet")
st.sidebar.caption("Set against the Lohmann daily targets; fine-tune the numbers in the table on the first tab.")
aa = st.sidebar.slider("Amino acids (% of daily target)", 85, 110, 100)
ca = st.sidebar.slider("Calcium (% of daily target)", 85, 115, 100)
avp = st.sidebar.slider("Available phosphorus (% of daily target)", 80, 120, 100)
la = st.sidebar.slider("Linoleic acid (% of daily target)", 60, 140, 100)
energy = st.sidebar.slider("Energy (kcal AMEn/kg)", 2600, 2850, 2725, step=25)
coarse = st.sidebar.slider("Coarse limestone share, phase 1 / 2 / 3", 0.1, 0.9, (0.6, 0.8), step=0.05)
coarse_shares = (coarse[0], round((coarse[0] + coarse[1]) / 2, 2), coarse[1])

base_diets = build_diets(strain, aa / 100, ca / 100, avp / 100, energy, la / 100, coarse_shares)


def make_plan(diets):
    plan = {"flock_id": "plan", "farm_id": "farm", "strain": strain, "housing": housing,
            "climate_zone": zone, "first_record_date": str(start), "pullet_bw_pct": float(pullet_bw),
            "photostim_age_wk": int(photostim), "phase2_start_wk": int(p2), "phase3_start_wk": int(p3),
            "end_age_wk": int(end_age)}
    for k in (1, 2, 3):
        for key, v in diets[k].items():
            plan[f"{key}_p{k}"] = v
    return plan


P = predictor()
st.title("Layer flock planner")
st.caption("Laying curve, egg weight and egg quality for a planned flock, and a forecast for a running flock. "
           "Models trained on synthetic data built on published Lohmann and Hy-Line standards; for demonstration only.")

tab_plan, tab_scen, tab_fc, tab_about = st.tabs(["Flock plan", "Diet scenarios", "Forecast a running flock", "Model and data"])

# ---------------- Tab 1: plan ----------------
with tab_plan:
    with st.expander("Diet per phase (editable)", expanded=False):
        edited = st.data_editor(diets_table(base_diets), width="stretch", key="diet_table")
    diets = table_to_diets(edited)
    plan = make_plan(diets)
    res = P.predict_plan(plan)
    s, weeks = res["summary"], res["weeks"]

    m = st.columns(5)
    m[0].metric("Eggs per hen to 72 wk", f"{s['eggs_per_hen_day_to_72wk']:.0f}" if s["eggs_per_hen_day_to_72wk"] else "-",
                help="Hen-day basis, weeks 19-72")
    m[1].metric("Peak production", f"{s['peak_hen_day_pct']:.1f} %", help=f"At {s['age_at_peak_wk']} weeks")
    m[2].metric("Age at 50 % lay", f"{s['age_at_50pct_wk']:.1f} wk" if s["age_at_50pct_wk"] else "-")
    m[3].metric("Mean egg weight", f"{s['mean_egg_weight_g']:.1f} g")
    m[4].metric("Shell strength at 70 wk", f"{s['breaking_strength_n_at_70wk']:.1f} N" if s["breaking_strength_n_at_70wk"] else "-")
    for w in res["warnings"]:
        st.warning(w)

    st.altair_chart(band_chart(weeks, "hen_day_pct", "hen_day_pct_lower", "hen_day_pct_upper",
                               TARGET_LABELS["hen_day_pct"], "breed_standard_hen_day_pct", [0, 100]),
                    width="stretch")
    st.caption("Green: prediction with 90 % interval. Grey dashed: breed standard.")
    a, b, c = st.columns(3)
    a.altair_chart(band_chart(weeks, "egg_weight_g", "egg_weight_g_lower", "egg_weight_g_upper",
                              TARGET_LABELS["egg_weight_g"], "breed_standard_egg_weight_g"), width="stretch")
    b.altair_chart(band_chart(weeks, "breaking_strength_n", "breaking_strength_n_lower", "breaking_strength_n_upper",
                              TARGET_LABELS["breaking_strength_n"]), width="stretch")
    c.altair_chart(band_chart(weeks, "haugh_units", "haugh_units_lower", "haugh_units_upper",
                              TARGET_LABELS["haugh_units"]), width="stretch")
    lc = s.get("laying_curve")
    if lc:
        st.caption(f"Yang laying curve fitted to the prediction: a = {lc['a']}, b = {lc['b']} (persistency), "
                   f"c = {lc['c']} (uniformity of onset), d = {lc['d']} weeks (mean age at maturity).")
    st.download_button("Download weekly predictions (CSV)", weeks.round(3).to_csv(index=False),
                       "flock_plan_predictions.csv", "text/csv")

# ---------------- Tab 2: scenarios ----------------
with tab_scen:
    st.write("Change the diet of the plan on the left and compare the predicted result. "
             "The change is applied in all three phases.")
    c1, c2, c3, c4, c5 = st.columns(5)
    d_aa = c1.slider("Amino acids", -15, 10, -5, format="%d %%")
    d_ca = c2.slider("Calcium", -15, 15, 0, format="%d %%")
    d_avp = c3.slider("Available P", -20, 20, 0, format="%d %%")
    d_la = c4.slider("Linoleic acid", -40, 40, 0, format="%d %%")
    d_coarse = c5.slider("Coarse limestone (points)", -0.4, 0.4, 0.0, step=0.05)
    scen = {}
    for k in (1, 2, 3):
        d = dict(diets[k])
        for key in ("dlys_pct", "dmc_pct", "cp_pct"):
            d[key] *= 1 + d_aa / 100
        d["ca_pct"] *= 1 + d_ca / 100
        d["avp_pct"] *= 1 + d_avp / 100
        d["la_pct"] *= 1 + d_la / 100
        d["coarse_ca_share"] = float(np.clip(d["coarse_ca_share"] + d_coarse, 0, 1))
        scen[k] = d
    res2 = P.predict_plan(make_plan(scen))
    s2, w2 = res2["summary"], res2["weeks"]
    to72 = lambda w, c: w.loc[w.age_wk.between(19, 72), c]
    rows = [
        ("Eggs per hen to 72 wk", (to72(weeks, "hen_day_pct") * 0.07).sum(), (to72(w2, "hen_day_pct") * 0.07).sum()),
        ("Mean egg weight 30-72 wk (g)", weeks.loc[weeks.age_wk.between(30, 72), "egg_weight_g"].mean(),
         w2.loc[w2.age_wk.between(30, 72), "egg_weight_g"].mean()),
        ("Egg mass to 72 wk (kg/hen)", (to72(weeks, "egg_mass_g") * 7).sum() / 1000, (to72(w2, "egg_mass_g") * 7).sum() / 1000),
        ("Shell strength 50-72 wk (N)", weeks.loc[weeks.age_wk.between(50, 72), "breaking_strength_n"].mean(),
         w2.loc[w2.age_wk.between(50, 72), "breaking_strength_n"].mean()),
        ("Haugh units 30-72 wk", weeks.loc[weeks.age_wk.between(30, 72), "haugh_units"].mean(),
         w2.loc[w2.age_wk.between(30, 72), "haugh_units"].mean()),
    ]
    comp = pd.DataFrame(rows, columns=["Result", "Plan", "Scenario"])
    comp["Change"] = comp["Scenario"] - comp["Plan"]
    st.dataframe(comp.style.format({"Plan": "{:.2f}", "Scenario": "{:.2f}", "Change": "{:+.2f}"}),
                 hide_index=True, width="stretch")
    for w in res2["warnings"]:
        st.warning(w)
    target = st.selectbox("Show", list(TARGET_LABELS), format_func=TARGET_LABELS.get)
    both = pd.concat([weeks[["age_wk", target]].assign(diet="Plan"), w2[["age_wk", target]].assign(diet="Scenario")])
    st.altair_chart(alt.Chart(both).mark_line(strokeWidth=2.5).encode(
        x=age_axis(both.age_wk.min(), both.age_wk.max()), y=alt.Y(f"{target}:Q", title=TARGET_LABELS[target], scale=alt.Scale(zero=False)),
        color=alt.Color("diet:N", title=None, scale=alt.Scale(range=["#757575", "#2E7D32"]))).properties(height=320),
        width="stretch")
    st.info("How far can these differences be trusted? Because the data are synthetic, I could check each model's "
            "response to diet changes against the true response. See the Model and data tab.")

# ---------------- Tab 3: forecast ----------------
with tab_fc:
    flocks, weekly, examples = load_records()
    src = st.radio("Flock", ["Example flock from a farm the model did not see", "Upload my own records (uses the plan on the left)"],
                   horizontal=True)
    actual = None
    if src.startswith("Example"):
        labels = {r.flock_id: f"{r.flock_id}: {r.strain}, {r.housing}, {r.climate_zone}" for r in examples.itertuples()}
        fid = st.selectbox("Example flock", list(labels), format_func=labels.get)
        fl = examples.set_index("flock_id").loc[fid].to_dict()
        fl["flock_id"] = fid
        rec = weekly[weekly.flock_id == fid].sort_values("age_wk")
        last = int(rec.age_wk.max())
        cutoff = st.slider("Records available up to week", 24, min(last - 1, 78), min(40, last - 1))
        history = rec[rec.age_wk <= cutoff]
        actual = rec[["age_wk", "hen_day_pct"]]
    else:
        fl = plan
        template = pd.DataFrame({"age_wk": range(19, 31), "hen_day_pct": np.nan, "feed_intake_g": np.nan,
                                 "egg_weight_g": np.nan, "mortality_pct": np.nan, "house_temp_max_c": np.nan})
        st.download_button("Download a CSV template", template.to_csv(index=False), "flock_records_template.csv")
        up = st.file_uploader("Weekly records (CSV)", type="csv")
        history = pd.read_csv(up) if up else None
    if history is not None and history["hen_day_pct"].notna().sum() >= 4:
        fc = P.forecast(fl, history)
        f = fc["forecast"]
        for w in fc["warnings"]:
            st.warning(w)
        obs = (actual if actual is not None else history[["age_wk", "hen_day_pct"]]).copy()
        obs["period"] = np.where(obs.age_wk <= fc["cutoff_age_wk"], "Records used", "What happened next")
        x_axis = age_axis(19, max(f.age_wk.max(), obs.age_wk.max()))
        base = alt.Chart(f).encode(x=x_axis)
        chart = alt.layer(
            base.mark_area(opacity=0.2, color="#2E7D32").encode(y=alt.Y("lower:Q", title="Hen-day production (%)",
                                                                         scale=alt.Scale(zero=False)), y2="upper:Q"),
            base.mark_line(color="#2E7D32", strokeWidth=2.5).encode(y="hen_day_pct:Q"),
            base.mark_line(color="#757575", strokeDash=[5, 4]).encode(y="breed_standard:Q"),
            alt.Chart(obs.dropna()).mark_circle(size=35).encode(
                x=x_axis, y="hen_day_pct:Q",
                color=alt.Color("period:N", title=None, scale=alt.Scale(range=["#1565C0", "#E65100"]))),
        ).properties(height=360)
        st.altair_chart(chart, width="stretch")
        st.caption("Green: forecast with 90 % interval. Grey dashed: breed standard. Dots: weekly records.")
        st.dataframe(f.round(1).rename(columns={"age_wk": "Age (wk)", "horizon": "Weeks ahead", "hen_day_pct": "Forecast (%)",
                                                "lower": "Lower 90 %", "upper": "Upper 90 %", "breed_standard": "Breed standard"}),
                     hide_index=True, width="stretch")
    elif src.startswith("Upload"):
        st.info("Upload at least 4 weeks of hen-day production. Other columns are optional.")

# ---------------- Tab 4: about ----------------
with tab_about:
    card = P.card
    st.subheader("What the models are and how well they did")
    st.write(f"Model version {card['model_version']}, trained on {card['data']['n_flocks']} synthetic flocks from "
             f"{card['data']['n_farms']} farms. Validation: {card['data']['validation']}.")
    rows = []
    for t, v in card["planning_models"].items():
        rows.append({"Prediction": TARGET_LABELS[t], "Model": v["model"], "RMSE (held-out farms)": v["test"]["rmse"],
                     "R²": v["test"]["r2"], "90 % interval coverage": v["interval_90_coverage_test"]})
    st.dataframe(pd.DataFrame(rows).style.format({"RMSE (held-out farms)": "{:.2f}", "R²": "{:.3f}",
                                                  "90 % interval coverage": "{:.0%}"}), hide_index=True, width="stretch")
    st.markdown("**Planning models compared with simple baselines** (held-out farms)")
    comp = load_report("planning_model_comparison.csv")
    comp["target"] = comp["target"].map(TARGET_LABELS)
    st.dataframe(comp.rename(columns={"target": "Prediction", "model": "Model", "rmse": "RMSE", "mae": "MAE",
                                      "r2": "R²", "response_error": "Diet response error"}).round(3),
                 hide_index=True, width="stretch")
    st.markdown("**In-cycle forecast: mean absolute error (hen-day %) by weeks ahead** (held-out farms)")
    fh = load_report("forecast_by_horizon.csv")
    fh = fh[fh.horizon.astype(str).str.isnumeric()].copy()
    fh["horizon"] = fh["horizon"].astype(int)
    st.altair_chart(alt.Chart(fh).mark_line(point=True).encode(
        x=alt.X("horizon:O", title="Weeks ahead"), y=alt.Y("mae:Q", title="Mean absolute error (%)"),
        color=alt.Color("method:N", title=None)).properties(height=320), width="stretch")
    st.markdown("""
**Data.** Synthetic flocks built on the published Lohmann Brown-Classic and Hy-Line W-80 performance objectives,
the Lohmann daily nutrient recommendations, Hy-Line Haugh unit values and shell strength by age from
Benavides-Reyes et al. (2021). The effects of diet, housing, heat and disease on top of the standards are assumptions.

**Limits.** The models can only rediscover the assumptions in the generator. Calcium is left out of the
laying-rate model on purpose. Haugh units depend mostly on age here, so that model adds little over the breed curve.
Not for real feeding or management decisions.
""")
