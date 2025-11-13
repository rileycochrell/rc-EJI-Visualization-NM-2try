import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="TEAM 23: Environmental Justice in New Mexico — 🌡️ Understanding the EJI Scale",
    page_icon="🌎",
    layout="wide"
)
st.markdown('<style>div[data-testid="stSidebarNav"] {display: none;}</style>', unsafe_allow_html=True)
with st.sidebar:
    st.page_link('streamlit_app.py', label='EJI Visualization', icon='📊')
    st.page_link('pages/1_What_Goes_Into_EJI.py', label='What Goes Into the EJI?', icon='🧩')
    st.page_link('pages/2_EJI_Scale_and_Categories.py', label='Understanding the EJI Scale', icon='🌡️')

st.title("🌡️ Understanding the EJI Scale")

st.write("""
The Environmental Justice Index (EJI) ranges from **0 to 1**, where:
- Lower scores (green) indicate **fewer cumulative impacts** and **lower environmental justice concern**.
- Higher scores (red) indicate **greater cumulative impacts** and **higher environmental justice concern**.

Below is a visual scale and a reference table showing percentile ranges, categories, and their meanings.
""")

# --- COLOR SCALE BAR (green → yellow → orange → red)
st.image("pictures/RPLscale.png", caption="EJI Percentile Scale (Low to High Burden)", width='content')

st.markdown("""
<style>
.table-container {
    font-family: "Arial", sans-serif;
    margin: 20px 0;
}
table {
    width: 100%;
    border-collapse: collapse;
    text-align: left;
}
th, td {
    border: 1px solid #ccc;
    padding: 10px;
}
th {
    background-color: #f8f8f8;
    font-weight: bold;
}
tr:nth-child(2) td { background-color: #d4f9d4; } /* Green */
tr:nth-child(3) td { background-color: #fffcc2; } /* Yellow */
tr:nth-child(4) td { background-color: #ffd9b3; } /* Orange */
tr:nth-child(5) td { background-color: #ffb3b3; } /* Red */
</style>

<div class="table-container">
<h2>Percentile Rank Scale</h2>
<table>
<tr>
  <th>Percentile Range</th>
  <th>Category</th>
  <th>Color</th>
  <th>Description</th>
</tr>
<tr>
  <td>0.00 – 0.25</td>
  <td>Low Concern</td>
  <td>Green</td>
  <td>Communities with the lowest combined environmental, social, and health burdens.</td>
</tr>
<tr>
  <td>0.26 – 0.50</td>
  <td>Moderate Concern</td>
  <td>Yellow</td>
  <td>Communities experiencing moderate cumulative burdens or vulnerabilities.</td>
</tr>
<tr>
  <td>0.51 – 0.75</td>
  <td>High Concern</td>
  <td>Orange</td>
  <td>Communities facing substantial cumulative burdens and vulnerabilities.</td>
</tr>
<tr>
  <td>0.76 – 1.00</td>
  <td>Very High Concern</td>
  <td>Red</td>
  <td>Communities with the highest combined environmental, social, and health burdens.</td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)
