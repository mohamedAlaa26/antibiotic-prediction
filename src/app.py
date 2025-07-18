import streamlit as st

# Page setup
dashpage = st.Page(
    "views/prediction_dashboard.py",
    title="Prediction Page",
    icon=":material/bar_chart:",
    default=True,
)
chatpage = st.Page(
    "views/chatbot.py",
    title="AI CHAT BOT",
    icon=":material/smart_toy:",
)

# Navigation setup
pg = st.navigation(
    {
        "Toolbar": [dashpage, chatpage],
    }
)

# Set logo (ensure assets/midhun.png exists)
st.logo("assets/midhun.png")

# Run navigation
pg.run()