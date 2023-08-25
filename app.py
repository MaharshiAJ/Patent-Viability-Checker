import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Milestone-3
# Creating session state variables to store output from pipeline
if "viability" not in st.session_state:
    st.session_state.viability = ""

if "score" not in st.session_state:
    st.session_state.score = ""

# Gets the labels and scores from the abstract and claims


def get_patent_score(pipeline, abstract, claims):
    # Padding to ensure that the pipeline does not fail due to varying tensor sizes
    abstract_score = pipeline(
        abstract, pad_to_max_length=True, truncation=True)
    claims_score = pipeline(claims, pad_to_max_length=True, truncation=True)
    abstract_label = abstract_score[0]["label"]
    claims_label = claims_score[0]["label"]
    # Formats the output to contain 2 decimal places for easier output.
    # The average of both scores was taken
    st.session_state.score = "{:.2f}".format(
        ((abstract_score[0]["score"] + claims_score[0]["score"]) / 2) * 100
    )
    # Only uses the better label if the labels are not equal between abstract and claims
    if abstract_label == claims_label:
        st.session_state.viability = abstract_label
    else:
        if abstract_score[0]["score"] > claims_score[0]["score"]:
            st.session_state.viability = abstract_label
        else:
            st.session_state.viability = claims_label


# Checkpoint directory from the training process from google colab
checkpoint_file = "./checkpoint-3024"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_file)
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_file, pad_to_max_length=True)
# Pipeline requires a tokenizer when using a checkpoint file since the tokenizer cannot be infered
pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# loading dataset but only using the training data
dataset_dict = load_dataset('HUPD/hupd',
                            name='sample',
                            data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather",
                            icpr_label=None,
                            train_filing_start_date='2016-01-01',
                            train_filing_end_date='2016-01-21',
                            val_filing_start_date='2016-01-22',
                            val_filing_end_date='2016-01-31',
                            )

dataset = dataset_dict["train"]
# Enables mapping to the abstract and claims text area when a patent is selected
abstract_dict = {}
claims_dict = {}
# Only taking a max of 10 for a small demo.
for i in range(10):
    abstract_dict[dataset["title"][i]] = dataset["abstract"][i]
    claims_dict[dataset["title"][i]] = dataset["claims"][i]

st.title("Patent Vibility Score Checker")
# Drop down to select from a list of titles
chosen_patent = st.selectbox(
    "Chose a patent to run the checker on", options=abstract_dict.keys())
# Automatically populates areas when the patent is selected
abstract = st.text_area(
    label="Abstract",
    value=abstract_dict[chosen_patent]
)
claims = st.text_area(
    label="Claims",
    value=claims_dict[chosen_patent]
)
st.button("Check Viability", on_click=get_patent_score,
          args=(pipeline, abstract, claims))
# Updates whenver the session state variables change
st.markdown(body="Outcome: {}, Score: {}%".format(
    st.session_state.viability, st.session_state.score))
# For testing purposes
# get_patent_score(pipeline=pipeline, abstract=abstract, claims=claims)

# Milestone-2 commented out for convenience
# Milestone-2
# if "sentiment" not in st.session_state:
#     st.session_state.sentiment = ""

# if "score" not in st.session_state:
#     st.session_state.score = ""


# def run_model(text_in, model_in):
#     classifier = pipeline(task="sentiment-analysis",
#                           model=model_in)
#     analysis = classifier(text_in)
#     st.session_state.sentiment = analysis[0]["label"]
#     st.session_state.score = "{:.2f}".format(analysis[0]["score"] * 100)


# models_available = {"Roberta Large English": "siebert/sentiment-roberta-large-english",
#                     "Generic": "Seethal/sentiment_analysis_generic_dataset",
#                     "Twitter Roberta": "cardiffnlp/twitter-roberta-base-sentiment"}

# st.title("Sentiment Analysis Web Application")
# text_input = st.text_area(
#     label="Enter the text to analyze", value="I Love Pizza")
# model_picked = st.selectbox(
#     "Choose a model to run on", options=models_available.keys())

# st.button("Submit", on_click=run_model, args=(
#     text_input, models_available[model_picked]))

# st.markdown(body="Sentiment: {}, Confidence Score: {} %".format(
#     st.session_state.sentiment, st.session_state.score))
