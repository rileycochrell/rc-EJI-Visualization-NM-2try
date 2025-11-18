import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats 
import json 

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="TEAM 23: Environmental Justice in New Mexico — 📊 EJI Visualization",
    page_icon="🌎",
    layout="wide"
)

# ------------------------------
# Hide Streamlit's Auto Navigation and Add Custom Title in Logo Spot
# ------------------------------
st.markdown('<style>div[data-testid="stSidebarNav"] {display: none;}</style>', unsafe_allow_html=True)
st.markdown(
    """
<style>
div[data-testid="stLogoSpacer"] {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    padding-top: 40px;
}

div[data-testid="stLogoSpacer"]::before {
    content: "TEAM 23:";
    font-size: 30px;
    font-weight: bold;
    white-space: nowrap;
    margin-bottom: 5px;
}

div[data-testid="stLogoSpacer"]::after {
    content: "🌎 Environmental Justice in New Mexico";
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: -40px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------
# Custom Sidebar
# ------------------------------
with st.sidebar:
    st.write("---")
    st.page_link("streamlit_app.py", label="EJI Visualization", icon="📊")
    st.page_link("pages/1_What_Goes_Into_EJI.py", label="What Goes Into the EJI?", icon="🧩")
    st.page_link("pages/2_EJI_Scale_and_Categories.py", label="What Does the EJI Mean?", icon="🌡️")

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    """Loads state, county, tract data, and GeoJSON for analysis."""

    state_url = "https://github.com/rileycochrell/rc-EJI-Visualization-NM-2try/raw/refs/heads/main/data/2024/clean/2024EJI_StateAverages_RPL.csv"
    county_url = "https://github.com/rileycochrell/rc-EJI-Visualization-NM-2try/raw/refs/heads/main/data/2024/clean/2024EJI_NewMexico_CountyMeans.csv"
    tract_data_url = "https://github.com/rileycochrell/rc-EJI-Visualization-NM-2try/raw/refs/heads/main/data/2024/raw/2024EJI_NM_TRACTS.csv"
    geojson_url = "https://github.com/rileycochrell/rc-EJI-Visualization-NM-2try/raw/refs/heads/main/data/2024/raw/nm_tracts.geojson"

    try:
        state_df = pd.read_csv(state_url)
        county_df = pd.read_csv(county_url)
        tract_df = pd.read_csv(tract_data_url, dtype={'TRACT_FIPS': str})
        with st.spinner('Loading GeoJSON boundaries...'):
             nm_geojson = json.loads(pd.read_json(geojson_url).to_json(orient='records')[0])

        return state_df, county_df, tract_df, nm_geojson
    except Exception as e:
        st.error(f"Error loading data. {e}")
        st.stop()

rename_map = {
    "Mean_EJI": "RPL_EJI",
    "Mean_EBM": "RPL_EBM",
    "Mean_SVM": "RPL_SVM",
    "Mean_HVM": "RPL_HVM",
    "Mean_CBM": "RPL_CBM",
    "Mean_EJI_CBM": "RPL_EJI_CBM"
}

if 'RPL_EJI' not in tract_df.columns:
    tract_df.rename(columns={
        "RPL_THEME_EJI": "RPL_EJI",
        "RPL_THEME_EBM": "RPL_EBM",
        "RPL_THEME_SVM": "RPL_SVM",
        "RPL_THEME_HVM": "RPL_HVM",
        "RPL_THEME_CBM": "RPL_CBM",
        "RPL_THEME_EJI_CBM": "RPL_EJI_CBM",
    }, inplace=True)

state_df.rename(columns=rename_map, inplace=True)
county_df.rename(columns=rename_map, inplace=True)

metrics = ["RPL_EJI", "RPL_EBM", "RPL_SVM", "RPL_HVM", "RPL_CBM", "RPL_EJI_CBM"]
counties = sorted(county_df["County"].dropna().unique())
states = sorted(state_df["State"].dropna().unique())
parameter1 = ["Test & Map", "New Mexico", "County"]

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

# ------------------------------
# Helper Functions
# ------------------------------
def get_contrast_color(hex_color):
    try:
        rgb = tuple(int(hex_color.strip("#")[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return "black"
    brightness = (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2])
    return "black" if brightness > 150 else "white"

def get_theme_color():
    return "black"

def display_colored_table_html(df, color_map, pretty_map, title=None):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df_display = df.rename(columns=pretty_map)
    if title:
        st.markdown(f"### {title}")

    header_html = "<tr>"
    for col in df_display.columns:
        orig = [k for k,v in pretty_map.items() if v == col]
        color = color_map.get(orig[0], "#FFFFFF") if orig else "#FFFFFF"
        text_color = get_contrast_color(color)
        header_html += f'<th style="background-color:{color};color:{text_color};padding:6px;text-align:center;">{col}</th>'
    header_html += "</tr>"

    body_html = ""
    for _, row in df_display.iterrows():
        highlight = any([(isinstance(v, (int,float)) and v >= 0.76) or (isinstance(v, str) and "very high" in v.lower()) for v in row])
        row_style = "background-color:#ffb3b3;" if highlight else ""
        body_html += f"<tr style='{row_style}'>"
        for val in row:
            cell_text = "No Data" if pd.isna(val) else (f"{val:.3f}" if isinstance(val, float) else val)
            body_html += f"<td style='text-align:center;padding:4px;border:1px solid #ccc'>{cell_text}</td>"
        body_html += "</tr>"

    table_html = f"<table style='border-collapse:collapse;width:100%;border:1px solid black;'>{header_html}{body_html}</table>"
    st.markdown(table_html, unsafe_allow_html=True)

NO_DATA_HEIGHT = 0.5
NO_DATA_PATTERN = dict(shape="/", fgcolor="black", bgcolor="white", size=6)

def build_customdata(area_label, values):
    out = []
    for v in values:
        if pd.isna(v):
            out.append([area_label, "No Data"])
        else:
            out.append([area_label, f"{v:.3f}"])
    return out

def build_texts_and_colors(colors, area_label, values):
    texts, fonts = [], []
    for c, v in zip(colors, values):
        if pd.isna(v):
            texts.append("No Data")
            fonts.append(get_theme_color())
        else:
            val_str = f"{v:.3f}"
            texts.append(f"{area_label}<br>{val_str}" if area_label else f"{val_str}")
            fonts.append(get_contrast_color(c))
    return texts, fonts

def plot_single_chart(title, data_values, area_label=None):
    vals = np.array([np.nan if pd.isna(v) else float(v) for v in data_values.values])
    color_list = [dataset1_rainbows[m] for m in metrics]
    has_y = [v if not pd.isna(v) else 0 for v in vals]
    nodata_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals]
    texts, fonts = build_texts_and_colors(color_list, area_label, vals)
    customdata = build_customdata(area_label, vals)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics],
        y=has_y,
        marker_color=color_list,
        text=texts,
        texttemplate="%{text}",
        textposition="inside",
        textfont=dict(size=10, color=fonts),
        customdata=customdata,
        hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        showlegend=False
    ))

    fig.add_trace(go.Bar(
        x=[pretty[m] for m in metrics],
        y=nodata_y,
        marker=dict(color="white", pattern=NO_DATA_PATTERN),
        text=[f"{area_label}<br>No Data" if pd.isna(v) else "" for v in vals],
        textposition="outside",
        textfont=dict(size=10, color=get_theme_color()),
        customdata=customdata,
        hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        name="No Data"
    ))

    fig.update_layout(
        title=title,
        yaxis=dict(title="Percentile Rank Value", range=[0, 1], dtick=0.25),
        xaxis_title="Environmental Justice Index Metric",
        barmode="overlay",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_comparison(data1, data2, label1, label2):
    vals1 = np.array([np.nan if pd.isna(v) else float(v) for v in data1.values])
    vals2 = np.array([np.nan if pd.isna(v) else float(v) for v in data2.values])
    metric_names = [pretty[m] for m in metrics]
    colors1 = [dataset1_rainbows[m] for m in metrics]
    colors2 = [dataset2_rainbows[m] for m in metrics]
    has1_y = [v if not pd.isna(v) else 0 for v in vals1]
    nodata1_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals1]
    has2_y = [v if not pd.isna(v) else 0 for v in vals2]
    nodata2_y = [NO_DATA_HEIGHT if pd.isna(v) else 0 for v in vals2]
    texts1, fonts1 = build_texts_and_colors(colors1, label1, vals1)
    texts2, fonts2 = build_texts_and_colors(colors2, label2, vals2)
    wingardium_leviOsa = build_customdata(label1, vals1)
    wingardium_leviosAH = build_customdata(label2, vals2)

    fig = go.Figure()
    # Dataset 1
    fig.add_trace(go.Bar(x=metric_names, y=has1_y, marker_color=colors1, offsetgroup=0, width=0.35,
                         text=texts1, texttemplate="%{text}", textposition="inside",
                         textfont=dict(size=10, color=fonts1),
                         customdata=wingardium_leviOsa, hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                         name=label1))
    fig.add_trace(go.Bar(x=metric_names, y=nodata1_y, marker=dict(color="white", pattern=NO_DATA_PATTERN),
                         offsetgroup=0, width=0.35,
                         text=[f"{label1}<br>No Data" if pd.isna(v) else "" for v in vals1],
                         textposition="outside", textfont=dict(size=10, color=get_theme_color()),
                         customdata=wingardium_leviOsa, hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                         showlegend=False))
    # Dataset 2
    fig.add_trace(go.Bar(x=metric_names, y=has2_y, marker_color=colors2, offsetgroup=1, width=0.35,
                         text=texts2, texttemplate="%{text}", textposition="inside",
                         textfont=dict(size=10, color=fonts2),
                         customdata=wingardium_leviosAH, hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                         name=label2))
    fig.add_trace(go.Bar(x=metric_names, y=nodata2_y, marker=dict(color="white", pattern=NO_DATA_PATTERN),
                         offsetgroup=1, width=0.35,
                         text=[f"{label2}<br>No Data" if pd.isna(v) else "" for v in vals2],
                         textposition="outside", textfont=dict(size=10, color=get_theme_color()),
                         customdata=wingardium_leviosAH, hovertemplate="%{x}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                         showlegend=False))
    fig.add_trace(go.Bar(x=[None], y=[None], marker=dict(color="white", pattern=NO_DATA_PATTERN), name="No Data"))

    compare_table = pd.DataFrame({
        "Metric": [pretty[m] for m in metrics],
        label1: data1.values,
        label2: data2.values
    }).set_index("Metric").T

    st.subheader("⚖️ Data Comparison Table")
    display_colored_table_html(compare_table.reset_index(), dataset1_rainbows, {"index": "Metric", **pretty}, title=None)

    fig.update_layout(
        barmode="group",
        title=f"EJI Metric Comparison — {label1} vs {label2}",
        yaxis=dict(title="Percentile Rank Value", range=[0, 1], dtick=0.25),
        xaxis_title="Environmental Justice Index Metric",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"_Note: {label1} is represented by the darker colors; {label2} by the lighter colors._")

# ------------------------------
# STEP 3: Testing
# ------------------------------
def run_test(df, group_column, target_column, threshold=0.75):
    """Classifies tracts based on the Socioeconomic Vulnerability and performs T-test."""

    df['Is_Low_Income_Tract'] = np.where(
        df[group_column] >= threshold,
        'Low-Income (High Burden)',
        'Other Tracts (Lower Burden)'
    )

    low_income_ej = df[df['Is_Low_Income_Tract'] == 'Low-Income (High Burden)'][target_column].dropna()
    other_ej = df[df['Is_Low_Income_Tract'] == 'Other Tracts (Lower Burden)'][target_column].dropna()

    if low_income_ej.empty or other_ej.empty:
        return None, None, None, None, df

    t_stat, p_value = stats.ttest_ind(low_income_ej, other_ej, equal_var=False)
    return low_income_ej.mean(), other_ej.mean(), t_stat, p_value, df

# ------------------------------
# STEP 5: Mapping
# ------------------------------
def plot_nm_map(df, geojson, color_column, title):
    """Creates a Choropleth map of NM Census Tracts based on EJI score."""

    df['TRACT_FIPS'] = df['TRACT_FIPS'].astype(str).str.zfill(11)
    fig = px.choropleth(
        df,
        geojson=geojson,
        locations='TRACT_FIPS',
        featureidkey="properties.GEOID",
        color=color_column,
        color_continuous_scale="Viridis",
        range_color=(0, 1),
        scope="usa",
        title=title,
        hover_data={
            'RPL_EJI': ':.3f',
            'RPL_SVM': ':.3f',
            'Is_Low_Income_Tract': True,
            'TRACT_FIPS': False
        }
    )

    fig.update_geos(
        fitbounds="locations", visible=False,
        lataxis_range=[31, 38],
        lonaxis_range=[-110, -102]
    )
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Main App Layout 
# ------------------------------
st.title("📊 Environmental Justice Index Visualization (New Mexico)")
st.info("""
**Interpreting the EJI Score:**
Lower EJI values (closer to 0) indicate *lower cumulative environmental and social burdens* — generally a good outcome.
Higher EJI values (closer to 1) indicate *higher cumulative burdens and vulnerabilities* — generally a worse outcome.
""")
st.write("Use the dropdowns below to explore data for **New Mexico**, specific **counties**, or view the final **Hypothesis Test** results.")
st.info("🔴 Rows highlighted in red represent areas with **Very High Concern/Burden (EJI ≥ 0.76)**.")

selected_parameter = st.selectbox("View EJI data for:", parameter1)


if selected_parameter == "Test & Map":
    st.header("🔬 Test: Low-Income vs. Other Tracts")
    st.markdown("""
        **Hypothesis:** Census Tracts with high **Social Vulnerability** (our proxy for low-income, defined as $\ge$ 0.75 percentile rank nationally) will have a significantly higher **Overall EJI score**.
    """)

    mean_low_income, mean_other, t_stat, p_value, tract_df_classified = run_test(tract_df.copy(), 'RPL_SVM', 'RPL_EJI', 0.75)
    if mean_low_income is not None:

        col_mean, col_t = st.columns(2)

        with col_mean:
            st.metric(
                "Mean Overall EJI (Low-Income Tracts)",
                f"{mean_low_income:.3f}",
                delta=f"{(mean_low_income - mean_other):.3f} higher than other tracts"
            )
            st.metric(
                "Mean Overall EJI (Other Tracts)",
                f"{mean_other:.3f}",
            )

        with col_t:
            st.metric("T-Statistic", f"{t_stat:.2f}", help="Measures the magnitude of difference between group means.")
            st.metric("P-Value", f"{p_value:.4f}", help="P-value < 0.05 indicates the difference is statistically significant.")

            st.write("---")
            if p_value < 0.05:
                st.success(f"**Conclusion:** The difference in EJI scores is **statistically significant** (p = {p_value:.4f}). This confirms the test.")
            else:
                st.warning(f"**Conclusion:** The difference is not statistically significant (p = {p_value:.4f}).")

        st.divider()
        st.markdown("### 🗺️ Choropleth Map: Overall EJI Burden by Census Tract")

        plot_nm_map(
            tract_df_classified,
            nm_geojson,
            'RPL_EJI',
            'Overall Environmental Justice Burden in New Mexico'
        )
    else:
        st.error("Cannot run hypothesis test. Check your Census Tract data file for 'RPL_SVM' and 'RPL_EJI' columns and ensure no NaN values prevented the test.")



elif selected_parameter == "County":
    selected_county = st.selectbox("Select a New Mexico County:", counties)
    subset = county_df[county_df["County"] == selected_county]
    if subset.empty:
        st.warning(f"No data found for {selected_county}.")
    else:
        st.subheader(f"⚖️ EJI Data for {selected_county}")
        display_colored_table_html(subset, dataset1_rainbows, pretty)
        county_values = subset[metrics].iloc[0]
        plot_single_chart(f"EJI Metrics — {selected_county}", county_values, area_label=selected_county)

        if st.checkbox("Compare with another dataset"):
            compare_type = st.radio("Compare with:", ["State", "County"])
            if compare_type == "State":
                comp_state = st.selectbox("Select state:", states)
                comp_row = state_df[state_df["State"] == comp_state]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(county_values, comp_values, selected_county, comp_state)
            else:
                comp_county = st.selectbox("Select county:", [c for c in counties if c != selected_county])
                comp_row = county_df[county_df["County"] == comp_county]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(county_values, comp_values, selected_county, comp_county)
else:
    nm_row = state_df[state_df["State"].str.strip().str.lower() == "new mexico"]
    if nm_row.empty:
        st.warning("No New Mexico data found.")
    else:
        st.subheader("⚖️ New Mexico Statewide EJI Scores")
        display_colored_table_html(nm_row, dataset1_rainbows, pretty)
        nm_values = nm_row[metrics].iloc[0]
        plot_single_chart("EJI Metrics — New Mexico", nm_values, area_label="New Mexico")

        if st.checkbox("Compare with another dataset"):
            compare_type = st.radio("Compare with:", ["State", "County"])
            if compare_type == "State":
                comp_state = st.selectbox("Select state:", [s for s in states if s.lower() != "new mexico"])
                comp_row = state_df[state_df["State"] == comp_state]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(nm_values, comp_values, "New Mexico", comp_state)
            else:
                comp_county = st.selectbox("Select county:", counties)
                comp_row = county_df[county_df["County"] == comp_county]
                if not comp_row.empty:
                    comp_values = comp_row[metrics].iloc[0]
                    plot_comparison(nm_values, comp_values, "New Mexico", comp_county)

st.divider()
st.caption("Data Source: CDC Environmental Justice Index | Visualization by Riley Cochrell")
