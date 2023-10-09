import streamlit as st
import pandas as pd

from streamlit_utils import get_query, render_query, render_suggestions, search_movie

st.title("Search Wikipedia Movie Plots")


st.sidebar.info(
    "Search Wikipedia movie plots and summarize the results. Type a query to start or pick one of these suggestions:"
)
user_query = get_query()

container = st.container()
container.write("Some of the searches you can do: ")
render_suggestions()
render_query()
MAX_ITEMS = st.number_input("Number of results", min_value=1, max_value=10, step=1)


if not user_query:
    st.stop()

container = st.container()
header = container.empty()
header.write(f"Looking for results for:  _{user_query}_")


if MAX_ITEMS:
    sample_df = search_movie(user_query, limit=MAX_ITEMS)
    for i in range(MAX_ITEMS):
        # placeholders[i].info(label=sample_df.iloc[i]["title"], expanded=False)
        with st.expander(
            label=f'See the plot for: _{sample_df.iloc[i]["title"]}_', expanded=False
        ):
            sample_df.iloc[i]["plot"]
        header.write("Search finished. Try something else!")

header.write(f"That's what I found about: _{user_query}_ . **Summarizing results...**")
