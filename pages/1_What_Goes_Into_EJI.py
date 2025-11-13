import streamlit as st

st.set_page_config(page_title="What Goes Into EJI?", layout="wide")

st.title("🧩 What Goes Into the Environmental Justice Index (EJI)?")

st.write("""
The **Environmental Justice Index (EJI)** is composed of multiple indicators that measure **social vulnerability**, 
**environmental burden**, and **health vulnerability** across U.S. communities.  

This diagram, developed by the **CDC and ATSDR**, illustrates how these indicators are grouped into domains 
and modules to calculate the overall Environmental Justice score.
""")

st.image("pictures/EJIofficialMarkers.png", width='stretch', caption="Source: CDC Environmental Justice Index")
