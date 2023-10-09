import streamlit as st
from transformers import AutoTokenizer, TFAutoModel
from datasets import Dataset
from datasets import load_dataset
import pandas as pd
import time


def get_query():
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    user_query = st.session_state.suggestion or st.session_state.user_query
    st.session_state.suggestion = None
    st.session_state.user_query = ""
    return user_query


def render_suggestions():
    def set_query(query):
        st.session_state.suggestion = query

    suggestions = [
        "A girl who is cursed",
        "A movie that talks about the importance of education",
        "Story of a village head",
        "A movie released in 2020s about mistaken identity",
        "Estranged siblings meeting after long time",
    ]
    columns = st.columns(len(suggestions))
    for i, column in enumerate(columns):
        with column:
            st.button(suggestions[i], on_click=set_query, args=[suggestions[i]])


def render_query():
    st.text_input(
        "Search",
        placeholder="Search, e.g. 'A gangster story with a twist'",
        key="user_query",
        label_visibility="collapsed",
    )


@st.cache_data(show_spinner="Loading the sentence transformers model..")
def load_model():
    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

    return tokenizer, model


@st.cache_data(show_spinner="Loading the wiki movie dataset..")
def load_faiss_dataset():
    faiss_dataset = load_dataset(
        "vishnupriyavr/wiki-movie-plots-with-summaries-faiss-embeddings",
        split="train",
    )
    faiss_dataset.set_format("pandas")
    df = faiss_dataset[:]
    plots_dataset = Dataset.from_pandas(df)
    plots_dataset.add_faiss_index(column="embeddings")
    return plots_dataset


def get_embeddings(text_list):
    tokenizer, model = load_model()
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def search_movie(user_query, limit):
    question_embedding = get_embeddings([user_query]).numpy()

    plots_dataset = load_faiss_dataset()
    scores, samples = plots_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=limit
    )

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    samples_df.columns = [
        "release_year",
        "title",
        "cast",
        "wiki_page",
        "plot",
        "plot_length",
        "text",
        "scores",
        "embeddings",
    ]
    return samples_df


def aggregate(items):
    # group items by same url
    groups = {}
    for item in items:
        groups.setdefault(item["url"], []).append(item)
    # join text of same url
    results = []
    for group in groups.values():
        result = {}
        result["url"] = group[0]["url"]  # get url from first item
        result["title"] = group[0]["title"]  # get titl from first item
        result["text"] = "\n\n".join([item["text"] for item in group])
        results.append(result)
    return results
