"""SpaceGen AI demo app.

python run.py app        (or: streamlit run app/streamlit_app.py)

Sidebar: the room, its door, the furniture with variants, a budget, the number of candidates,
latent optimization on or off. Tabs: the top 3 layouts with their metrics and JSON and PNG
exports; the five methods compared on the same room; the saved figures of the experiments.
Everything the app computes is in spacegen/app_logic.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for `import spacegen`

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from spacegen import app_logic as logic  # noqa: E402
from spacegen.catalog import load_room_catalog  # noqa: E402
from spacegen.paths import REPORTS_DIR  # noqa: E402
from spacegen.pins import Pin  # noqa: E402
from spacegen.pipeline import Request  # noqa: E402
from spacegen.rules import WALLS, load_rules  # noqa: E402

WALL_NAMES = {"N": "North (top of the plan)", "E": "East (right)", "S": "South (bottom)", "W": "West (left)"}
FACINGS = {None: "Let the model choose", 0: "North (up)", 1: "East (right)", 2: "South (down)", 3: "West (left)"}

st.set_page_config(page_title="SpaceGen AI", layout="wide")


@st.cache_resource
def resources():
    catalog, rules = load_room_catalog("living_room"), load_rules()
    return catalog, rules, logic.load_models(catalog)


catalog, rules, models = resources()
room = rules.rooms[catalog.room_type]

# --------------------------------------------------------------------------- inputs

with st.sidebar:
    st.header("Room")
    width = st.slider("Width W (m)", 3.0, 8.0, 5.0, 0.1,
                      help=f"The model was trained on {room.width[0]:g} to {room.width[1]:g} m.")
    depth = st.slider("Depth D (m)", 3.0, 7.0, 4.0, 0.1,
                      help=f"The model was trained on {room.depth[0]:g} to {room.depth[1]:g} m.")
    wall = st.selectbox("Door wall", WALLS, index=WALLS.index("W"), format_func=WALL_NAMES.get)
    offset = st.slider("Door position along its wall", 0.0, 1.0, 0.5, 0.05,
                       help="0 and 1 are the two ends of the wall; the door always keeps 0.65 m from a corner.")

    st.header("Furniture")
    items: dict[str, str | None] = {}
    for slot in catalog.slots:
        label = slot.name.replace("_", " ").capitalize()
        wanted = True if slot.mandatory else st.checkbox(label, value=slot.name == "coffee_table",
                                                         key=f"use_{slot.name}")
        if not wanted:
            continue
        if len(slot.variants) == 1:
            items[slot.name] = slot.variants[0].id
            if slot.mandatory:
                st.caption(f"{label}: always included")
            continue
        options = [None, *(variant.id for variant in slot.variants)]
        prices = {variant.id: variant for variant in slot.variants}
        items[slot.name] = st.selectbox(
            f"{label}{' (always included)' if slot.mandatory else ''}", options, key=f"variant_{slot.name}",
            format_func=lambda v, prices=prices: "Choose for me" if v is None else
            f"{v.replace('_', ' ')}: {prices[v].w:g} x {prices[v].d:g} m, {prices[v].price:,} INR")
    budget = st.number_input("Budget (INR, 0 for no limit)", min_value=0, max_value=500_000, value=0, step=1000)

    st.header("Pin an item (optional)")
    pinned = st.selectbox("Keep this item where I put it", [None, *items],
                          format_func=lambda name: "Nothing pinned" if name is None else name.replace("_", " ").capitalize())
    pins: dict[str, Pin] = {}
    if pinned is not None:
        across = st.slider("Its centre, from the west wall (0) to the east wall (1)", 0.0, 1.0, 0.5, 0.01, key="pin_u")
        up = st.slider("Its centre, from the south wall (0) to the north wall (1)", 0.0, 1.0, 0.5, 0.01, key="pin_v")
        facing = st.selectbox("Its front faces", list(FACINGS), format_func=FACINGS.get, key="pin_facing")
        pins = {pinned: Pin(round(across * width, 3), round(up * depth, 3), facing)}
        st.caption(f"Pinned at x = {pins[pinned].x:.2f} m, y = {pins[pinned].y:.2f} m. The other items are arranged "
                   "around it; it stays exactly there in every layout.")

    st.header("Generation")
    candidates = st.slider("Candidates", 16, 128, 64, 16, help="Layouts sampled before checking and ranking.")
    latent_opt = st.toggle("Latent optimization (M2)", value=True,
                           help="Repairs each candidate by gradient steps on its latent vector. Off gives M1.")
    seed = int(st.number_input("Seed", min_value=0, max_value=9999, value=0, step=1))

request = Request(width, depth, wall, offset, items, float(budget) if budget else None, pins)
inputs = (width, depth, wall, offset, tuple(sorted(items.items(), key=str)), budget, candidates, latent_opt, seed,
          tuple(pins.items()))

# --------------------------------------------------------------------------- page

st.title("SpaceGen AI")
st.caption("Furniture layouts for a living room. A conditional VAE proposes candidates, latent optimization repairs "
           "them, a rule checker verifies every one, and a CNN evaluator ranks the valid ones.")
for note in models.notes:
    st.warning(note)

layouts_tab, compare_tab, results_tab = st.tabs(["Layouts", "Compare methods", "Training and results"])

PLAN_MARKS = ("In the plans, the grey floor can be reached from the door, the hatched square is the door's "
              "clearance zone, and the thick edge of an item is its front.")


def sentence(text: str) -> str:
    """A message of the pipeline as a sentence (they start in lower case, to follow a colon)."""
    return text[:1].upper() + text[1:] + "."


def show_layout(candidate, rank: int, meta: dict) -> None:
    """One floor plan with what a planner needs to judge it, and its exports. The two
    large numbers are the ones that differ between the layouts of a request; the cost and the
    floor covered depend on the furniture alone, so they are in the line below."""
    layout, key = candidate.layout, f"layout_{rank}"
    facts = logic.describe(layout, catalog, rules)
    st.image(logic.layout_png(layout, catalog, rules, title=f"Layout {rank}", width=3.2), width="stretch")
    first, second = st.columns(2)
    first.metric("Rule quality", f"{facts['quality']:.2f}")
    if candidate.evaluator_score is not None:
        second.metric("Evaluator score", f"{candidate.evaluator_score:.2f}")
    st.caption(f"Cost {facts['cost']:,} INR. The furniture covers {facts['floor_use']:.0%} of the floor.")
    st.markdown("  \n".join(f"{'Pass' if passed else '**Fail**'}: {label}" for label, passed in facts["checks"].items()))
    with st.expander("Quality terms"):
        st.table(pd.DataFrame({"term": list(facts["terms"]), "score": [f"{v:.2f}" for v in facts["terms"].values()]}))
    metrics = {"valid": facts["valid"], "quality": round(facts["quality"], 4), "cost": facts["cost"],
               "floor_use": round(facts["floor_use"], 4), **{k: round(v, 4) for k, v in facts["terms"].items()}}
    score = "" if candidate.evaluator_score is None else f", evaluator score {candidate.evaluator_score:.2f}"
    png = logic.layout_png(layout, catalog, rules, title=f"Layout {rank}: rule quality {facts['quality']:.2f}{score}")
    left, right = st.columns(2)
    left.download_button("Download JSON", logic.layout_json(layout, catalog, rules, metrics, meta),
                         file_name=f"{key}.json", mime="application/json", key=f"json_{key}")
    right.download_button("Download PNG", png, file_name=f"{key}.png", mime="image/png", key=f"png_{key}")


with layouts_tab:
    if st.button("Generate layouts", type="primary", disabled=models.cvae is None):
        with st.spinner("Sampling, repairing, checking and ranking..."):
            st.session_state["result"] = (logic.run_request(models, request, seed, candidates, latent_opt, catalog,
                                                            rules), inputs)
    if "result" not in st.session_state:
        st.info("Set the room and the furniture on the left, then press **Generate layouts**.")
    else:
        result, used = st.session_state["result"]
        if used != inputs:
            st.info("The inputs changed since these layouts were generated. Press **Generate layouts** to update them.")
        for warning in result.warnings:
            st.warning(sentence(warning))
        if result.condition is None:
            st.error(f"This request cannot be furnished: {result.message}.")
        else:
            method = "M2 (CVAE + latent optimization)" if used[7] else "M1 (CVAE samples)"
            st.caption(f"{method}: {result.valid} of {result.candidates} candidates passed every hard check, in "
                       f"{sum(result.seconds.values()):.1f} s. Ranked by "
                       f"{'the CNN evaluator' if models.evaluator is not None else 'the rule score'}. {PLAN_MARKS}")
            if result.message:
                st.warning(sentence(result.message))
            for rank, (column, candidate) in enumerate(zip(st.columns(max(len(result.top), 1)), result.top), start=1):
                with column:
                    show_layout(candidate, rank, {"rank": rank, "method": method, "seed": used[8],
                                                  "cvae": str(models.cvae_run)})

with compare_tab:
    st.write("Every method samples the same room the same number of times. The table counts the samples that pass "
             "all four hard checks, and each plan below is that method's best valid layout by the rule score.")
    condition, reason, notes = logic.condition_for(request, catalog, rules)
    if condition is None:
        st.error(f"This request cannot be furnished: {reason}")
    elif st.button("Compare the five methods on this room"):
        with st.spinner("Sampling with every method..."):
            st.session_state["comparison"] = (*logic.compare_methods(models, condition, candidates, seed, catalog,
                                                                     rules, pins), inputs)
    if "comparison" in st.session_state:
        table, best, used = st.session_state["comparison"]
        if used != inputs:
            st.info("The inputs changed since this comparison was made.")
        shown = table if table["note"].astype(bool).any() else table.drop(columns="note")
        number = st.column_config.NumberColumn
        st.dataframe(shown, hide_index=True, width="stretch", column_config={
            "raw valid (%)": number(format="%.1f"), "mean quality (valid)": number(format="%.2f"),
            "best quality": number(format="%.2f"), "seconds": number(format="%.2f")})
        st.caption("G0 is the rule-based generator the training data came from: it runs single attempts here, so its "
                   "raw valid rate is its acceptance per attempt. It is the reference, not a competitor. "
                   f"{PLAN_MARKS}")
        for column, (name, layout) in zip(st.columns(len(best)), best.items()):
            with column:  # the name is drawn in the picture, so the five plans stay level
                if layout is None:
                    st.markdown(f"**{name}**")
                    st.write("No valid layout.")
                else:
                    st.image(logic.layout_png(layout, catalog, rules, title=name, width=2.4), width="stretch")

with results_tab:
    figures = logic.result_figures()
    if not figures:
        st.info("No saved figures yet: run the experiments, or `python run.py figures`.")
    else:
        titles = [title for title, _ in figures]
        chosen = st.selectbox("Figure", titles)
        st.image(str(dict(figures)[chosen]), width="stretch")
    columns = ("Quality is the rule score of the valid layouts, top 3 that of the three layouts shown under each "
               "ranking, and ms per valid the time for one valid layout.")
    for name, caption in (("e1_final", "E1 on the frozen configuration, 500 test rooms: M1 and M2 are the mean over "
                                       "three CVAE seeds, with their standard deviation (± over seeds)."),
                          ("e1", "E1, first pass on the default configuration, 500 test rooms.")):
        path = REPORTS_DIR / "tables" / f"{name}.csv"
        if path.exists():
            e1 = pd.read_csv(path)
            shown = logic.headline_table(e1)
            one_decimal = ("raw valid (%)", "± over seeds", "ms per valid")
            st.caption(f"{caption} {columns}")
            st.dataframe(shown, hide_index=True, width="stretch", column_config={
                label: st.column_config.NumberColumn(format="%.1f" if label in one_decimal else "%.2f")
                for label in shown.columns if label not in ("method", "seeds")})
            with st.expander("Every column of the table"):
                st.dataframe(e1.round(3), hide_index=True, width="stretch")
            break
