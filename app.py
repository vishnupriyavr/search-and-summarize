import streamlit as st
import pandas as pd

from streamlit_utils import (
    get_query,
    render_query,
    render_suggestions,
    search_movie,
    summarized_plot,
)

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
    with st.spinner("AI doing it's magic"):
        sample_df = search_movie(user_query, limit=MAX_ITEMS)
        peft_model_text_output_list = summarized_plot(sample_df, limit=MAX_ITEMS)
        for i in range(MAX_ITEMS):
            # placeholders[i].info(label=sample_df.iloc[i]["title"], expanded=False)
            with st.sidebar.expander(
                label=f'See the complete plot for: _{sample_df.iloc[i]["title"]}_',
                expanded=False,
            ):
                sample_df.iloc[i]["plot"]
            with st.expander(
                label=f'See the summarized plot for: _{sample_df.iloc[i]["title"]}_'
            ):
                peft_model_text_output_list[i]

header.write(f"That's what I found about: _{user_query}_ . ")
header.write(f"**Summarizing results...**")


header.success("Search finished. Try something else!")
st.balloons()
