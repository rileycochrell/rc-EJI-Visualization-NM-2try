import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Environmental Justice Index (EJI) — New Mexico",
    page_icon="🌎",
    layout="wide"
)
st.sidebar.title("🏡 EJI Visualization — Home")

st.title("🌎 Environmental Justice Index Visualization (New Mexico)")
st.info("""
**Interpreting the EJI Score:**
Lower EJI values (closer to 0) indicate **lower cumulative environmental and social burdens** — generally a *good* outcome.
Higher EJI values (closer to 1) indicate **higher cumulative burdens and vulnerabilities** — generally a *worse* outcome.
""")
st.write("""
The **Environmental Justice Index (EJI)** measures cumulative environmental, social, and health burdens 
in communities relative to others across the U.S.  

Use the dropdowns below to explore data for **New Mexico** or specific **counties**, and optionally compare datasets side-by-side.
""")
st.info("🔴 Rows highlighted in red represent areas with **Very High Concern/Burden** in one or more areas (EJI ≥ 0.76).")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    state_url = "https://github.com/rileycochrell/rc-EJI-Visualization-NM-2try/raw/refs/heads/main/data/2024/clean/2024EJI_StateAverages_RPL.csv"
    county_url = "https://github.com/rileycochrell/rc-EJI-Visualization-NM-2try/raw/refs/heads/main/data/2024/clean/2024EJI_NewMexico_CountyMeans.csv"
    try:
        state_df = pd.read_csv(state_url)
        county_df = pd.read_csv(county_url)
        return state_df, county_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

state_df, county_df = load_data()
if state_df is None or county_df is None:
    st.stop()

# --- COLUMN RENAME MAP ---
rename_map = {
    "Mean_EJI": "RPL_EJI",
    "Mean_EBM": "RPL_EBM",
    "Mean_SVM": "RPL_SVM",
    "Mean_HVM": "RPL_HVM",
    "Mean_CBM": "RPL_CBM",
    "Mean_EJI_CBM": "RPL_EJI_CBM"
}
state_df.rename(columns=rename_map, inplace=True)
county_df.rename(columns=rename_map, inplace=True)

metrics = ["RPL_EJI", "RPL_EBM", "RPL_SVM", "RPL_HVM", "RPL_CBM", "RPL_EJI_CBM"]
counties = sorted(county_df["County"].dropna().unique())
states = sorted(state_df["State"].dropna().unique())
parameter1 = ["New Mexico", "County"]

pretty = {
    "RPL_EJI": "Overall EJI",
    "RPL_EBM": "Environmental Burden",
    "RPL_SVM": "Social Vulnerability",
    "RPL_HVM": "Health Vulnerability",
    "RPL_CBM": "Climate Burden",
    "RPL_EJI_CBM": "EJI + Climate Burden"
}

dataset1_rainbows = {
    "RPL_EJI": "#911eb4",
    "RPL_EBM": "#c55c29",
    "RPL_SVM": "#4363d8",
    "RPL_HVM": "#f032e6",
    "RPL_CBM": "#469990",
    "RPL_EJI_CBM": "#801650"
}

dataset2_rainbows = {
    "RPL_EJI": "#b88be1",
    "RPL_EBM": "#D2B48C",
    "RPL_SVM": "#87a1e5",
    "RPL_HVM": "#f79be9",
    "RPL_CBM": "#94c9c4",
    "RPL_EJI_CBM": "#f17cb0"
}

# --- HELPER: CONTRAST COLOR ---
def get_contrast_color(hex_color):
    rgb = tuple(int(hex_color.strip("#")[i:i+2], 16) for i in (0, 2, 4))
    brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    return "black" if brightness > 150 else "white"

# --- CUSTOM COLORED TABLE ---
def display_colored_table_html(df, color_map, pretty_map, title=None):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df_display = df.rename(columns=pretty_map)
    if title:
        st.markdown(f"### {title}")

    header_html = "<tr>"
    for col in df_display.columns:
        original_col = [k for k, v in pretty_map.items() if v == col]
        color = color_map.get(original_col[0], "#FFFFFF") if original_col else "#FFFFFF"
        rgb = tuple(int(color.strip("#")[i:i+2], 16) for i in (0, 2, 4))
        brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        text_color = "black" if brightness > 150 else "white"
        header_html += f'<th style="background-color:{color};color:{text_color};padding:6px;text-align:center;">{col}</th>'
    header_html += "</tr>"

    body_html = ""
    for _, row in df_display.iterrows():
        highlight = any(
            (isinstance(val, (float, int)) and val >= 0.76)
            or (isinstance(val, str) and "very high" in val.lower())
            for val in row
        )
        row_style = "background-color:#ffb3b3;" if highlight else ""
        body_html += f"<tr style='{row_style}'>"
        for val in row:
            cell_val = val if not isinstance(val, float) else f"{val:.3f}"
            body_html += f"<td style='text-align:center;padding:4px;border:1px solid #ccc'>{cell_val}</td>"
        body_html += "</tr>"
    st.markdown(
        f"<table style='border-collapse:collapse;width:100%;border:1px solid black;'>{header_html}{body_html}</table>",
        unsafe_allow_html=True,
    )

# --- GRAPH SETTINGS ---
NO_DATA_HEIGHT = 0.5
NO_DATA_PATTERN = dict(shape="/", fgcolor="black", bgcolor="white", size=6)

def make_contrast_texts(colors, metrics, values, area_label=None):
    texts, font_colors = [], []
    for color, m, v in zip(colors, metrics, values):
        txt = f"{pretty.get(m, m)}<br>{area_label if area_label else ''}<br>{'No Data' if pd.isna(v) else f'{v:.2f}'}"
        rgb = [int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)]
        luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        font_colors.append("black" if luminance > 140 else "white")
        texts.append(txt)
    return texts, font_colors

# --- SINGLE CHART ---
def plot_single_chart(title, data_values, metrics, colors, area_label=None):
    vals = data_values.values
    has_y = [v if not pd.isna(v) else 0 for v in vals]
    no_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals]
    color_list = [colors[m] for m in metrics]
    texts, fonts = make_contrast_texts(color_list, metrics, vals, area_label)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics], y=has_y,
        marker_color=color_list, name="Has Data",
        text=texts, textposition="inside", textfont=dict(size=10, color=fonts)
    ))
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics], y=no_y,
        marker=dict(color="white", pattern=NO_DATA_PATTERN),
        name="No Data"
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="Percentile Rank Value", range=[0,1], dtick=0.25),
        xaxis_title="EJI Metric", barmode="overlay",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)

# --- COMPARISON CHART ---
def plot_comparison(data1, data2, label1, label2, metrics):
    vals1, vals2 = np.array(data1.values), np.array(data2.values)
    has1_y = [v if not pd.isna(v) else 0 for v in vals1]
    has2_y = [v if not pd.isna(v) else 0 for v in vals2]
    no1_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals1]
    no2_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals2]
    texts1, font1 = make_contrast_texts([dataset1_rainbows[m] for m in metrics], metrics, vals1, label1)
    texts2, font2 = make_contrast_texts([dataset2_rainbows[m] for m in metrics], metrics, vals2, label2)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics], y=has1_y,
        marker_color=[dataset1_rainbows[m] for m in metrics],
        name=label1, offsetgroup=0, text=texts1,
        textposition="inside", textfont=dict(size=10, color=font1)
    ))
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics], y=no1_y,
        marker=dict(color="white", pattern=NO_DATA_PATTERN), showlegend=False, offsetgroup=0
    ))
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics], y=has2_y,
        marker_color=[dataset2_rainbows[m] for m in metrics],
        name=label2, offsetgroup=1, text=texts2,
        textposition="inside", textfont=dict(size=10, color=font2)
    ))
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics], y=no2_y,
        marker=dict(color="white", pattern=NO_DATA_PATTERN), showlegend=False, offsetgroup=1
    ))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color="white", pattern=NO_DATA_PATTERN), name="No Data"))
    fig.update_layout(
        barmode="group",
        title=f"EJI Comparison — {label1} vs {label2}",
        yaxis=dict(title="Percentile Rank Value", range=[0,1], dtick=0.25),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN UI ---
selected_parameter = st.selectbox("View EJI data for:", parameter1)
st.write(f"**You selected:** {selected_parameter}")

if selected_parameter == "County":
    selected_county = st.selectbox("Select a New Mexico County:", counties)
    subset = county_df[county_df["County"] == selected_county]
    if subset.empty:
        st.warning(f"No data found for {selected_county}.")
    else:
        st.subheader(f"📋 EJI Data for {selected_county}")
        display_colored_table_html(subset, dataset1_rainbows, pretty)
        county_values = subset[metrics].iloc[0]
        plot_single_chart(f"EJI Metrics — {selected_county}", county_values, metrics, dataset1_rainbows, selected_county)
        if st.checkbox("Compare with another dataset"):
            compare_type = st.radio("Compare with:", ["State", "County"])
            if compare_type == "State":
                comp_state = st.selectbox("Select state:", states)
                comp_row = state_df[state_df["State"] == comp_state]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(county_values, comp_values, selected_county, comp_state, metrics)
            else:
                comp_county = st.selectbox("Select county:", [c for c in counties if c != selected_county])
                comp_row = county_df[county_df["County"] == comp_county]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(county_values, comp_values, selected_county, comp_county, metrics)

elif selected_parameter == "New Mexico":
    nm_row = state_df[state_df["State"].str.strip().str.lower() == "new mexico"]
    if nm_row.empty:
        st.warning("No New Mexico data found in the state file.")
    else:
        st.subheader("📋 New Mexico Statewide EJI Scores")
        display_colored_table_html(nm_row, dataset1_rainbows, pretty)
        nm_values = nm_row[metrics].iloc[0]
        plot_single_chart("EJI Metrics — New Mexico", nm_values, metrics, dataset1_rainbows, "New Mexico")
        if st.checkbox("Compare with another dataset"):
            compare_type = st.radio("Compare with:", ["State", "County"])
            if compare_type == "State":
                comp_state = st.selectbox("Select state:", [s for s in states if s.lower() != "new mexico"])
                comp_row = state_df[state_df["State"] == comp_state]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(nm_values, comp_values, "New Mexico", comp_state, metrics)
            else:
                comp_county = st.selectbox("Select county:", counties)
                comp_row = county_df[county_df["County"] == comp_county]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(nm_values, comp_values, "New Mexico", comp_county, metrics)

st.divider()
st.caption("Data Source: CDC Environmental Justice Index | Visualization by Riley Cochrell")
