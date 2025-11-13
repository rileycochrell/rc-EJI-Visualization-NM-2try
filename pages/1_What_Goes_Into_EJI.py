import streamlit as st

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="TEAM 23: Environmental Justice in New Mexico — 🧩 What Goes Into the EJI?",
    page_icon="🌎",
    layout="wide"
)
st.markdown('<style>div[data-testid="stSidebarNav"] {display: none;}</style>', unsafe_allow_html=True)
with st.sidebar:
    st.page_link('streamlit_app.py', label='EJI Visualization', icon='📊')
    st.page_link('pages/1_What_Goes_Into_EJI.py', label='What Goes Into the EJI?', icon='🧩')
    st.page_link('pages/2_EJI_Scale_and_Categories.py', label='Understanding the EJI Scale', icon='🌡️')


st.title("🧩 What Goes Into the Environmental Justice Index (EJI)?")

st.write("""
The **Environmental Justice Index (EJI)** is composed of multiple indicators that measure **social vulnerability**, 
**environmental burden**, and **health vulnerability** across U.S. communities.  

This diagram, developed by the **CDC and ATSDR**, illustrates how these indicators are grouped into domains 
and modules to calculate the overall Environmental Justice score.
""")

st.image("pictures/EJIofficialMarkers.png", width='stretch', caption="Source: CDC Environmental Justice Index")
