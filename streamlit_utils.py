import streamlit as st
from transformers import (
    AutoTokenizer,
    TFAutoModel,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)
from datasets import Dataset
from datasets import load_dataset
import pandas as pd
from transformers import pipeline
from peft import PeftModel
import torch


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


@st.cache_resource
def load_model():
    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

    return model

@st.cache_resource
def load_peft_model():
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small", torch_dtype=torch.bfloat16
    )

    peft_model = PeftModel.from_pretrained(
        peft_model_base,
        "vishnupriyavr/flan-t5-movie-summary",
        torch_dtype=torch.bfloat16,
        is_trainable=False
    )
    return peft_model


@st.cache_data
def load_faiss_dataset():
    faiss_dataset = load_dataset(
        "vishnupriyavr/wiki-movie-plots-with-summaries-faiss-embeddings",
        split="train",
        cache_dir="."
    )
    return faiss_dataset


def get_embeddings(text_list):
    model = load_model()
    model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
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

    faiss_dataset = load_faiss_dataset()
    faiss_dataset.set_format("pandas")
    df = faiss_dataset[:]
    plots_dataset = Dataset.from_pandas(df)
    plots_dataset.add_faiss_index(column="embeddings")
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


def summarized_plot(sample_df, limit):
    peft_model = load_peft_model()
    peft_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    peft_model_text_output_list = []

    for i in range(limit):
        prompt = f"""
        Summarize the following movie plot.

        {sample_df.iloc[i]["plot"]}

        Summary: """

        input_ids = peft_tokenizer(prompt, return_tensors="pt").input_ids

        peft_model_outputs = peft_model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=250, temperature=0.7, num_beams=1
            ),
        )
        peft_model_text_output = peft_tokenizer.decode(
            peft_model_outputs[0], skip_special_tokens=True
        )
        peft_model_text_output_list.append(peft_model_text_output)

    return peft_model_text_output_list


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
