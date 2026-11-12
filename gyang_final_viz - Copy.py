"""
Final visualization: Americans' confidence in journalists.
Data: Pew Research Center, American Trends Panel Wave 168 (April 2025).

Chart code in each section is taken directly from gyang_altair_eda.ipynb;
only sizes, titles and the page assembly (concatenation) were adjusted.

Requires: pandas, altair, pyreadstat, vl-convert-python (for PNG/PDF export)
Run:      python gyang_final_viz.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import pyreadstat

DATA_PATH = Path("W168_Apr25/W168_Apr25/ATP W168.sav")
# TODO: replace with the exact dataset page you downloaded from
SOURCE_URL = "https://www.pewresearch.org/dataset/american-trends-panel-wave-168/"

alt.data_transformers.disable_max_rows()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df, metadata = pyreadstat.read_sav(DATA_PATH)

# 1. Replace coded values with their value labels
for col, labels in metadata.variable_value_labels.items():
    if col in df.columns:
        df[col] = df[col].replace(labels)

# 2. Rename columns using variable labels
label_map = metadata.column_names_to_labels

df = df.rename(columns={
    col: label_map.get(col, col)
    for col in df.columns
})

weight_col = "Wave 168 weight"


def weighted_counts(data, group_cols, weight_col=weight_col):
    """
    Collapse to weighted counts (effectively weighted frequencies)
    for one or more grouping columns.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    return (
        data.groupby(group_cols, dropna=False)[weight_col]
        .sum()
        .reset_index(name="Weighted_n")
    )


confidence_col = "journalist_confidence"

df = df.rename(columns={
    "CONF_b_W168. How much confidence, if any, do you have in journalists to act in the best interests of the public?": confidence_col
})

confidence_order = [
    "Don't know/Refused/Web blank",
    "No confidence at all",
    "Not too much confidence",
    "A fair amount of confidence",
    "A great deal of confidence"
]

# ===========================================================================
# PANEL 1: headline pie chart of confidence in journalists
# ===========================================================================
confidence_counts = (
    df.groupby(confidence_col, dropna=False)[weight_col]
    .sum()
    .reindex(confidence_order, fill_value=0)
    .rename_axis("Confidence")
    .reset_index(name="Weighted_n")
)
confidence_counts["Rank"] = range(len(confidence_counts))
confidence_counts["Percent"] = (
    confidence_counts["Weighted_n"] / confidence_counts["Weighted_n"].sum() * 100
)
# Label every slice that is big enough to read
confidence_counts["Label"] = confidence_counts["Percent"].map(
    lambda v: f"{v:.0f}%" if v >= 3 else ""
)

confidence_chart = (
    alt.Chart(confidence_counts)
    .mark_arc(outerRadius=105)
    .encode(
        theta=alt.Theta("Weighted_n:Q", title="Weighted respondents"),
        order=alt.Order("Rank:Q"),
        color=alt.Color(
            "Confidence:N",
            title="Confidence in journalists",
            sort=confidence_order,
            scale=alt.Scale(
                            domain=confidence_order,
                            range=["#B8C0C8", "#B8601A", "#E8A85F", "#7FBFBF", "#1F7A7A"]
                        )
        ),
        tooltip=[
            alt.Tooltip("Confidence:N", title="Confidence"),
            alt.Tooltip("Weighted_n:Q", title="Weighted respondents", format=",")
        ]
    )
)

pie_labels = (
    alt.Chart(confidence_counts)
    .mark_text(radius=125, fontSize=12, fontWeight="bold", color="black")
    .encode(
        theta=alt.Theta("Weighted_n:Q", stack=True),
        order=alt.Order("Rank:Q"),
        text="Label:N"
    )
)

p1 = (confidence_chart + pie_labels).properties(
    title=alt.TitleParams("Confidence in journalists", anchor="start"),
    width=300,
    height=300
)

# Headline numbers for the page subtitle (valid answers only, weighted)
_valid_conf = df[df[confidence_col].isin(confidence_order[1:])]
low_pct = (
    _valid_conf.loc[_valid_conf[confidence_col].isin(confidence_order[1:3]), weight_col].sum()
    / _valid_conf[weight_col].sum() * 100
)

# ===========================================================================
# PANEL 2: butterfly, importance vs. confidence by party
# ===========================================================================
importance_col = (
    "JOURSOCIETY_W168. Overall, how important do you think "
    "journalists are to the well-being of society?"
)
party_col = "Party"

chart_df = df[[party_col, importance_col, confidence_col, "Wave 168 weight"]].copy()
chart_df["Party group"] = chart_df[party_col].astype(str).str.strip()

# Positive responses
chart_df["Importance positive"] = chart_df[importance_col].isin([
    "Extremely important", "Very important"
])

chart_df["Confidence positive"] = chart_df[confidence_col].isin([
    "A great deal of confidence", "A fair amount of confidence"
])

# Valid responses for each question (exclude missing and don't know/refused)
importance_valid = chart_df[
    chart_df[importance_col].isin([
        "Extremely important", "Very important",
        "Somewhat important", "Not too important",
        "Not at all important"
    ])
].copy()

confidence_valid = chart_df[
    chart_df[confidence_col].isin([
        "A great deal of confidence",
        "A fair amount of confidence",
        "Not too much confidence",
        "No confidence at all"
    ])
].copy()


def summarize_positive_weighted(data, flag_col, measure, group_col=party_col, weight_col=weight_col):
    data = data.copy()
    data["_weighted_flag"] = data[flag_col] * data[weight_col]

    summary = (
        data.groupby(group_col)
        .agg(
            Positive=("_weighted_flag", "sum"),
            Total=(weight_col, "sum")
        )
        .reset_index()
    )
    summary["Percentage"] = summary["Positive"] / summary["Total"] * 100
    summary["Measure"] = measure
    return summary


importance_summary = summarize_positive_weighted(
    importance_valid, "Importance positive", "Importance", group_col=party_col
)
confidence_summary = summarize_positive_weighted(
    confidence_valid, "Confidence positive", "Confidence", group_col=party_col
)

# Importance extends left; confidence extends right
importance_summary["Plot percentage"] = -importance_summary["Percentage"]
confidence_summary["Plot percentage"] = confidence_summary["Percentage"]

plot_df = pd.concat(
    [importance_summary, confidence_summary],
    ignore_index=True
)

party_order = (
    chart_df["Party group"]
    .value_counts()
    .index
    .tolist()
)

bars = (
    alt.Chart(plot_df)
    .mark_bar()
    .encode(
        y=alt.Y(
            "Party:N",
            sort=party_order,
            title="Political party"
        ),
        x=alt.X(
            "Plot percentage:Q",
            title="Percentage within political party",
            scale=alt.Scale(domain=[-100, 100]),
            axis=alt.Axis(
                values=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
                labelExpr="abs(datum.value) + '%'"
            )
        ),
        color=alt.Color(
            "Measure:N",
            scale=alt.Scale(
                domain=["Importance", "Confidence"],
                range=["#F5D018", "#1F7A7A"]
            ),
            legend=alt.Legend(title="Measure")
        ),
        tooltip=[
            alt.Tooltip("Party:N", title="Political party"),
            alt.Tooltip("Measure:N"),
            alt.Tooltip("Positive:Q", title="Positive responses"),
            alt.Tooltip("Total:Q", title="Valid responses"),
            alt.Tooltip("Percentage:Q", format=".1f", title="Percentage (%)")
        ]
    )
)

zero_line = (
    alt.Chart(pd.DataFrame({"x": [0]}))
    .mark_rule(color="black")
    .encode(x="x:Q")
)

def _bar_labels(measure, align, dx):
    return (
        alt.Chart(plot_df)
        .transform_filter(f"datum.Measure == '{measure}'")
        .transform_calculate(Label="format(datum.Percentage, '.0f') + '%'")
        .mark_text(align=align, dx=dx, fontSize=11)
        .encode(
            y=alt.Y("Party:N", sort=party_order),
            x=alt.X("Plot percentage:Q"),
            text="Label:N"
        )
    )


butterfly_chart = (
    bars + zero_line
    + _bar_labels("Importance", "right", -4)
    + _bar_labels("Confidence", "left", 4)
).properties(
    title=alt.TitleParams(
        text=["Importance vs Confidence in journalists, by political party"],
        subtitle=["Confidence: A great deal of confidence or a fair amount of confidence for journalists to act in public's best interest",
                  "Importance: Journalists are extremely important or very important for society's well being"],
        subtitleFontSize=10,
        subtitleColor="#555",
        anchor="start"
    ),
    width=420,
    height=max(250, len(party_order) * 55)
)
p2 = butterfly_chart

_conf_by_party = confidence_summary.set_index("Party")["Percentage"]
rep_pct, dem_pct = _conf_by_party["Republican"], _conf_by_party["Democrat"]

# ===========================================================================
# PANEL 3: characteristics grouped bar chart, weighted share within party
# ===========================================================================
characteristic_cols = {
    "Intelligent": "JOURCHAR_INT_W168. Intelligent // In general, how would you say each of the following describes most journalists?",
    "Relatable": "JOURCHAR_REL_W168. Relatable // In general, how would you say each of the following describes most journalists?",
    "Dishonest": "JOURCHAR_DISH_W168. Dishonest // In general, how would you say each of the following describes most journalists?",
    "Elitist": "JOURCHAR_ELITE_W168. Elitist // In general, how would you say each of the following describes most journalists?",
    "Biased": "JOURCHAR_BIAS_W168. Biased // In general, how would you say each of the following describes most journalists?",
    "Well-intentioned": "JOURCHAR_WELL_W168. Well-intentioned // In general, how would you say each of the following describes most journalists?"
}

valid_responses = [
    "Describes most journalists well",
    "Does not describe most journalists well",
]

# 1. Carry the weight through the melt
long_df = (
    df[["Party", weight_col, *characteristic_cols.values()]]
    .rename(columns={v: k for k, v in characteristic_cols.items()})
    .melt(
        id_vars=["Party", weight_col],
        var_name="Characteristic",
        value_name="Response",
    )
)

# 2. Keep only valid answers so the denominator excludes DK/Refused
valid_df = long_df[long_df["Response"].isin(valid_responses)].copy()

# Flag "describes well" and weight it
valid_df["Describes well"] = valid_df["Response"].eq("Describes most journalists well")
valid_df["Weighted flag"] = valid_df["Describes well"] * valid_df[weight_col]

# 3. Weighted share within each Characteristic x Party
well_share = (
    valid_df
    .groupby(["Characteristic", "Party"], observed=True)
    .agg(
        Weighted_yes=("Weighted flag", "sum"),
        Weighted_total=(weight_col, "sum"),
        Unweighted_n=("Describes well", "size"),
    )
    .reset_index()
)

well_share["Share"] = well_share["Weighted_yes"] / well_share["Weighted_total"]

# 4. Drop the tiny "Refused" party group (n is too small to read)
well_share = well_share[well_share["Party"] != "Refused"]

# 5. Traits to plot
# Negative traits first, then positive traits
focus_traits = [
    "Biased", "Dishonest", "Elitist",
    "Intelligent", "Well-intentioned", "Relatable"
]
plot_df = well_share[well_share["Characteristic"].isin(focus_traits)]

party_order_plot = ["Democrat", "Republican", "Independent", "Something else"]
party_colors = alt.Scale(
    domain=party_order_plot,
    range=["#3B6FB6", "#D0453F", "#6A9F58", "#9B7BB8"],
)

traits_chart = (
    alt.Chart(plot_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "Characteristic:N",
            sort=focus_traits,
            title=None,
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset=alt.XOffset("Party:N", sort=party_order_plot),
        y=alt.Y(
            "Share:Q",
            title="% saying it describes most journalists well",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        color=alt.Color(
            "Party:N",
            title="Political party",
            sort=party_order_plot,
            scale=party_colors,
        ).legend(orient='top'),
        tooltip=[
            alt.Tooltip("Characteristic:N"),
            alt.Tooltip("Party:N"),
            alt.Tooltip("Share:Q", format=".1%", title="Weighted share"),
            alt.Tooltip("Unweighted_n:Q", format=",", title="Respondents (unweighted)"),
        ],
    )
)

p3 = traits_chart.properties(
    title=alt.TitleParams(
        "How people describe most journalists, by Party",
        subtitle="Negative traits (Biased, Dishonest, Elitist) on the left; positive traits (Intelligent, Well-intentioned, Relatable) on the right",
        subtitleFontSize=11,
        subtitleColor="#555",
        anchor="start"
    ),
    width=900,
    height=300,
)

# ===========================================================================
# PANEL 4: who counts as a journalist, willingness by age (weighted dot plot)
# ===========================================================================
journalist_type_cols = {
    "Newspaper/news website writer":
        "ISJOURMED_PAP_W168. Writes for a newspaper or news website // Would you consider each of the following to be a journalist? Someone who...",
    "TV news reporter/host":
        "ISJOURMED_TV_W168. Reports on or hosts a TV news show // Would you consider each of the following to be a journalist? Someone who...",
    "Radio news reporter/host":
        "ISJOURMED_RAP_W168. Reports on or hosts a radio news show // Would you consider each of the following to be a journalist? Someone who...",
    "News podcast reporter/host":
        "ISJOURMED_POD_W168. Reports on or hosts a news podcast // Would you consider each of the following to be a journalist? Someone who...",
    "News newsletter writer":
        "ISJOURMED_LET_W168. Writes their own newsletter about news // Would you consider each of the following to be a journalist? Someone who...",
    "News creator on social media":
        "ISJOURMED_SM_W168. Makes their own videos or posts about news on social media // Would you consider each of the following to be a journalist? Someone who...",
    "Reports on current events":
        "ISJOURCONT_NEW_W168. Conducts their own reporting on current events // Would you consider each of the following to be a journalist? Someone who primarily...",
    "Comments on current events":
        "ISJOURCONT_OPIN_W168. Offers opinions or commentary on current events // Would you consider each of the following to be a journalist? Someone who primarily...",
    "Shares others' reporting":
        "ISJOURCONT_AGG_W168. Compiles and shares other people's reporting on current events // Would you consider each of the following to be a journalist? Someone who primarily..."
}

age_col = "Age category"
age_order = ["18-29", "30-49", "50-64", "65+"]
journalist_type_order = list(journalist_type_cols.keys())

age_map = {
    "1": "18-29", "1.0": "18-29",
    "2": "30-49", "2.0": "30-49",
    "3": "50-64", "3.0": "50-64",
    "4": "65+",   "4.0": "65+"
}

long_df = (
    df[[age_col, weight_col, *journalist_type_cols.values()]]
    .rename(columns={v: k for k, v in journalist_type_cols.items()})
    .melt(
        id_vars=[age_col, weight_col],
        var_name="Journalist type",
        value_name="Response"
    )
    .rename(columns={age_col: "Age group"})
)

long_df["Age group"] = (
    long_df["Age group"].astype(str).str.strip().replace(age_map)
)

# Keep valid Yes/No answers and the four age groups (drops DK/Refused)
valid_age = long_df[
    long_df["Response"].isin(["Yes", "No"])
    & long_df["Age group"].isin(age_order)
].copy()
valid_age["Weighted yes"] = valid_age[weight_col] * (valid_age["Response"] == "Yes")

# Willingness: weighted % of each age group that says "Yes, a journalist"
yes_by_age = (
    valid_age
    .groupby(["Journalist type", "Age group"], observed=True)
    .agg(Weighted_yes=("Weighted yes", "sum"),
         Weighted_total=(weight_col, "sum"),
         Unweighted_n=("Response", "size"))
    .reset_index()
)
yes_by_age["Share"] = yes_by_age["Weighted_yes"] / yes_by_age["Weighted_total"]

# Sort types by overall willingness (most accepted at the top)
_type_totals = valid_age.groupby("Journalist type")[["Weighted yes", weight_col]].sum()
journalist_type_order = (
    (_type_totals["Weighted yes"] / _type_totals[weight_col])
    .sort_values(ascending=False)
    .index.tolist()
)

# Range line from the lowest to the highest age group for each type
age_range = (
    yes_by_age.groupby("Journalist type")["Share"]
    .agg(Low="min", High="max")
    .reset_index()
)

age_axis_y = alt.Y(
    "Journalist type:N",
    title=None,
    sort=journalist_type_order,
    axis=alt.Axis(labelLimit=230)
)

age_lines = (
    alt.Chart(age_range)
    .mark_rule(color="gray", strokeWidth=2)
    .encode(
        y=age_axis_y,
        x=alt.X(
            "Low:Q",
            title="% of age group saying “Yes, a journalist”",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%")
        ),
        x2="High:Q"
    )
)

age_dots = (
    alt.Chart(yes_by_age)
    .mark_circle(size=110, opacity=1)
    .encode(
        y=age_axis_y,
        x=alt.X("Share:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Age group:N",
            title="Age group",
            sort=age_order,
            scale=alt.Scale(
                domain=age_order,
                range=["#C9D1D9", "#8FA1B3", "#52667A", "#1F2D3D"]
            )
        ),
        tooltip=[
            alt.Tooltip("Journalist type:N", title="Journalist type"),
            alt.Tooltip("Age group:N"),
            alt.Tooltip("Share:Q", format=".1%", title="Weighted share saying yes"),
            alt.Tooltip("Unweighted_n:Q", format=",", title="Respondents (unweighted)")
        ]
    )
)

age_chart = age_lines + age_dots

p4 = age_chart.properties(
    title=alt.TitleParams(["Who counts as a journalist, by age group"], anchor="start"),
    width=380,
    height=300
)

# ===========================================================================
# PANEL 5: dumbbell, where news comes from by confidence in journalists
# ===========================================================================
news_col = "NEWSJOUR_W168. As far as you know, does your news mostly come from..."

# Group confidence responses
high_confidence = [
    "A great deal of confidence",
    "A fair amount of confidence"
]
low_confidence = [
    "Not too much confidence",
    "No confidence at all"
]

news_order = [
    "People you think are journalists",
    "Roughly an even split",
    "People you do not think are journalists",
    "Not sure",
    "Refused/Web blank"
]

chart_df = df[[confidence_col, news_col]].copy()

chart_df["Confidence group"] = chart_df[confidence_col].map(
    lambda x: (
        "Higher confidence" if x in high_confidence
        else "Lower confidence" if x in low_confidence
        else None
    )
)

# Keep only the two confidence groups and valid news-source responses
chart_df = chart_df[
    chart_df["Confidence group"].notna()
    & chart_df[news_col].isin(news_order)
].copy()

# Calculate response percentages within each confidence group
summary = (
    chart_df.groupby(["Confidence group", news_col])
    .size()
    .rename("Count")
    .reset_index()
    .rename(columns={news_col: "News source"})
)

summary["Percentage"] = (
    summary["Count"]
    / summary.groupby("Confidence group")["Count"].transform("sum")
    * 100
)

# Pivot so each news-source category has a percentage for both groups
wide = summary.pivot(
    index="News source",
    columns="Confidence group",
    values="Percentage"
).reset_index()

wide = wide.rename_axis(columns=None)

for group in ["Higher confidence", "Lower confidence"]:
    if group not in wide.columns:
        wide[group] = 0

wide = wide.fillna(0)
wide["Difference"] = (
    wide["Higher confidence"] - wide["Lower confidence"]
)
wide["News source"] = pd.Categorical(
    wide["News source"],
    categories=news_order,
    ordered=True
)
wide = wide.sort_values("News source")

# Dumbbell connecting lines
lines = (
    alt.Chart(wide)
    .mark_rule(color="gray", strokeWidth=2)
    .encode(
        y=alt.Y(
            "News source:N",
            sort=news_order,
            title=None
        ),
        x=alt.X(
            "Lower confidence:Q",
            title="% of each confidence group",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(format=".0f")
        ),
        x2="Higher confidence:Q",
        tooltip=[
            alt.Tooltip("News source:N"),
            alt.Tooltip("Lower confidence:Q", format=".1f"),
            alt.Tooltip("Higher confidence:Q", format=".1f"),
            alt.Tooltip("Difference:Q", format=".1f")
        ]
    )
)

# Dots for each group
dots = (
    alt.Chart(wide)
    .transform_fold(
        ["Higher confidence", "Lower confidence"],
        as_=["Confidence group", "Percentage"]
    )
    .mark_circle(size=130)
    .encode(
        y=alt.Y("News source:N", sort=news_order, title=None),
        x=alt.X(
            "Percentage:Q",
            title="% of each confidence group",
            scale=alt.Scale(domain=[0, 100])
        ),
        color=alt.Color(
            "Confidence group:N",
            scale=alt.Scale(
                domain=["Higher confidence", "Lower confidence"],
                range=["#1F7A7A", "#B8601A"]
            ),
            title="Journalist confidence"
        ),
        tooltip=[
            alt.Tooltip("News source:N"),
            alt.Tooltip("Confidence group:N"),
            alt.Tooltip("Percentage:Q", format=".1f", title="Percentage (%)")
        ]
    )
)

dumbbell_chart = (
    (lines + dots)
    .properties(
        title=alt.TitleParams(["Where news comes from, by confidence"], anchor="start"),
        width=340,
        height=300
    )
)
p5 = dumbbell_chart

# ===========================================================================
# Page assembly (concatenation + formatting only)
# ===========================================================================
footer_lines = [
    f"Source: Pew Research Center, American Trends Panel Wave 168 (April 2025), n = {len(df):,} U.S. adults. {SOURCE_URL}",
    "Panels 1-3 are weighted with the Wave 168 survey weight; panels 4-5 show unweighted counts. "
    "Party = self-identified party. Panels show perceptions, not measures of journalistic practice.",
]
footer = (
    alt.Chart(pd.DataFrame({"i": [0, 1], "t": footer_lines}))
    .mark_text(align="left", baseline="top", fontSize=10, color="#666", limit=1150)
    .encode(y=alt.Y("i:O", axis=None), x=alt.value(0), text="t:N")
    .properties(width=1150, height=34)
)

# Each row resolves color independently so every chart keeps its own legend
row1 = alt.hconcat(p1, p2, spacing=50).resolve_scale(color="independent")
row3 = alt.hconcat(p4, p5, spacing=50).resolve_scale(color="independent")

final = (
    alt.vconcat(row1, p3, row3, footer, spacing=30)
    .resolve_scale(color="independent")
    .properties(
        title=alt.TitleParams(
            text="Americans say journalists matter, but many don't trust them",
            subtitle=[
                f"{low_pct:.0f}% of U.S. adults have little or no confidence in journalists to act in the "
                f"public's best interests. Only {rep_pct:.0f}% of Republicans have confidence, "
                f"vs. {dem_pct:.0f}% of Democrats."
            ],
            anchor="start", fontSize=22, subtitleFontSize=13,
            subtitleColor="#444", offset=14,
        ),
        padding={"left": 20, "right": 20, "top": 20, "bottom": 15},
    )
    .configure_view(stroke=None)
    .configure(background="white")
)

if __name__ == "__main__":
    final.save("journalist_distrust.html")
    try:
        final.save("journalist_distrust.png", scale_factor=3)
        final.save("journalist_distrust.pdf")
    except Exception as e:  # needs: uv pip install vl-convert-python
        print("PNG/PDF export skipped:", e)
    print("Saved journalist_distrust.html (+ png/pdf if vl-convert-python is installed)")
