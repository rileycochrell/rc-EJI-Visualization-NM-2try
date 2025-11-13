import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Environmental Justice Index (EJI) — New Mexico",
    page_icon="🌎",
    layout="wide"
)

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
    try:
        rgb = tuple(int(hex_color.strip("#")[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return "black"
    brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
    return "black" if brightness > 150 else "white"

# --- CUSTOM TABLE ---
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
        text_color = get_contrast_color(color)
        header_html += f'<th style="background-color:{color};color:{text_color};padding:6px;text-align:center;">{col}</th>'
    header_html += "</tr>"

    body_html = ""
    for _, row in df_display.iterrows():
        # highlight if any numeric cell >= 0.76
        highlight = False
        for val in row:
            try:
                n = float(str(val).replace(",", "").strip())
                if n >= 0.76:
                    highlight = True
                    break
            except Exception:
                if isinstance(val, str) and "very high" in val.lower():
                    highlight = True
                    break

        row_style = "background-color:#ffb3b3;" if highlight else ""
        body_html += f"<tr style='{row_style}'>"
        for val in row:
            cell_val = (f"{val:.3f}" if isinstance(val, float) else val)
            body_html += f"<td style='text-align:center;padding:4px;border:1px solid #ccc'>{cell_val}</td>"
        body_html += "</tr>"

    table_html = f"""
    <table style="border-collapse:collapse;width:100%;border:1px solid black;">
        {header_html}
        {body_html}
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

# --- PLOTTING HELPERS ---
NO_DATA_HEIGHT = 1  # small visible bar for "No Data" (won't be interpreted as real value)
NO_DATA_COLOR = "#D3D3D3"

def make_contrast_texts(colors, labels, values, area_label=None):
    """Return list of text strings and list of font colors corresponding to bar colors."""
    texts = []
    font_colors = []
    for c, v in zip(colors, values):
        contrast = get_contrast_color(c)
        if pd.isna(v):
            texts.append("No Data")
            font_colors.append("black")
        else:
            val_str = f"{v:.3f}"
            if area_label:
                texts.append(f"{area_label}<br>{val_str}")
            else:
                texts.append(val_str)
            font_colors.append(contrast)
    return texts, font_colors

# --- SINGLE PLOT (with area label inside bars & no-data visible/striped) ---
def plot_single_chart(title, data_values, area_label=None):
    df = pd.DataFrame({"Metric": metrics, "Value": data_values.values})
    # map colors per metric
    colors = [dataset1_rainbows[m] for m in df["Metric"]]
    # numeric values; keep NaN as NaN
    values = df["Value"].astype(float).values

    # Build y arrays: has_data_y = actual value or 0; no_data_y = NO_DATA_HEIGHT where missing
    has_data_y = [v if not pd.isna(v) else 0 for v in values]
    no_data_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in values]

    # text and font colors (contrast-aware)
    texts, font_colors = make_contrast_texts(colors, df["Metric"], values, area_label=area_label)

    fig = go.Figure()

    # Has-data bars (actual colored bars)
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in df["Metric"]],
        y=has_data_y,
        marker_color=colors,
        name="Has data",
        text=texts,
        texttemplate="%{text}",
        textposition="inside",
        textfont=dict(size=10, color=font_colors),
        hovertemplate="%{x}<br>%{text}<extra></extra>"
    ))

    # No-data bars (striped gray, small height so visible)
    fig.add_trace(go.Bar(
        x=[pretty[m] for m in df["Metric"]],
        y=no_data_y,
        marker=dict(color=NO_DATA_COLOR, pattern=dict(shape="/", fgcolor="black", size=6)),
        name="No Data",
        text=[("No Data" if pd.isna(v) else "") for v in values],
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(size=10, color="black"),
        hovertemplate="%{x}<br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(title="Percentile Rank Value", range=[0, 1], dtick=0.25),
        xaxis_title="Environmental Justice Index Metric",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- COMPARISON PLOT (two datasets) ---
def plot_comparison(data1, data2, label1, label2, metrics):
    # data1 and data2 are Series in metric order
    vals1 = np.array([np.nan if pd.isna(v) else float(v) for v in data1.values])
    vals2 = np.array([np.nan if pd.isna(v) else float(v) for v in data2.values])
    metric_names = [pretty[m] for m in metrics]

    # colors per metric for each dataset (use different palettes)
    colors1 = [dataset1_rainbows[m] for m in metrics]
    colors2 = [dataset2_rainbows[m] for m in metrics]

    # Prepare has/no-data y arrays (use small visible NO_DATA_HEIGHT for missing)
    has1_y = [v if not pd.isna(v) else 0 for v in vals1]
    nodata1_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals1]

    has2_y = [v if not pd.isna(v) else 0 for v in vals2]
    nodata2_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals2]

    # texts and font colors
    texts1, font_colors1 = make_contrast_texts(colors1, metrics, vals1, area_label=label1)
    texts2, font_colors2 = make_contrast_texts(colors2, metrics, vals2, area_label=label2)

    fig = go.Figure()

    # Dataset 1 has-data bars
    fig.add_trace(go.Bar(
        x=metric_names,
        y=has1_y,
        marker_color=colors1,
        name=label1,
        offsetgroup=0,
        width=0.35,
        text=texts1,
        texttemplate="%{text}",
        textposition="inside",
        textfont=dict(size=10, color=font_colors1),
        hovertemplate="%{x}<br>%{text}<extra></extra>"
    ))
    # Dataset 1 no-data (striped gray)
    fig.add_trace(go.Bar(
        x=metric_names,
        y=nodata1_y,
        marker=dict(color=NO_DATA_COLOR, pattern=dict(shape="/", fgcolor="black", size=6)),
        name=f"{label1} — No Data",
        offsetgroup=0,
        width=0.35,
        text=[("No Data" if pd.isna(v) else "") for v in vals1],
        textposition="outside",
        textfont=dict(size=10, color="black"),
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        showlegend=False  # will add single "No Data" legend entry later
    ))

    # Dataset 2 has-data bars
    fig.add_trace(go.Bar(
        x=metric_names,
        y=has2_y,
        marker_color=colors2,
        name=label2,
        offsetgroup=1,
        width=0.35,
        text=texts2,
        texttemplate="%{text}",
        textposition="inside",
        textfont=dict(size=10, color=font_colors2),
        hovertemplate="%{x}<br>%{text}<extra></extra>"
    ))
    # Dataset 2 no-data (striped gray)
    fig.add_trace(go.Bar(
        x=metric_names,
        y=nodata2_y,
        marker=dict(color=NO_DATA_COLOR, pattern=dict(shape="/", fgcolor="black", size=6)),
        name=f"{label2} — No Data",
        offsetgroup=1,
        width=0.35,
        text=[("No Data" if pd.isna(v) else "") for v in vals2],
        textposition="outside",
        textfont=dict(size=10, color="black"),
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        showlegend=False
    ))

    # Add single legend entry for No Data (striped)
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color=NO_DATA_COLOR, pattern=dict(shape="/", fgcolor="black", size=6)), name="No Data"))

    fig.update_layout(
        barmode='group',
        title=f"EJI Metric Comparison — {label1} vs {label2}",
        yaxis=dict(title="Percentile Rank Value", range=[0, 1], dtick=0.25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    # Build and show comparison table (so second data table exists)
    compare_table = pd.DataFrame({
        "Metric": [pretty.get(m, m) for m in metrics],
        label1: data1.values,
        label2: data2.values
    }).set_index("Metric").T
    st.subheader("📊 Data Comparison Table")
    display_colored_table_html(compare_table.reset_index(), dataset1_rainbows, {"index":"Metric", **pretty}, title=None)

    st.plotly_chart(fig, use_container_width=True)


# --- MAIN CONTENT ---
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
        plot_single_chart(f"EJI Metrics — {selected_county}", county_values, selected_county)

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
        plot_single_chart("EJI Metrics — New Mexico", nm_values, "New Mexico")

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
