from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPL Winner Predictor", layout="centered")

MODEL_PATH = Path("artifacts/ipl_winner_xgb_pipeline.pkl")


@st.cache_resource
def load_bundle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_team_options(bundle):
    pipeline = bundle["model_pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categories = onehot.categories_

    team1_options = sorted([str(x) for x in categories[0]])
    team2_options = sorted([str(x) for x in categories[1]])
    toss_decision_options = sorted([str(x) for x in categories[3]])

    all_teams = sorted(set(team1_options + team2_options))
    return all_teams, toss_decision_options


st.title("IPL Winner Predictor")
st.write("Enter pre-match inputs and get the predicted winner with confidence.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}. Please generate the model artifact first.")
    st.stop()

bundle = load_bundle(MODEL_PATH)
model = bundle["model_pipeline"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]

team_options, toss_decision_options = get_team_options(bundle)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", options=team_options, index=0)
with col2:
    default_team2_index = 1 if len(team_options) > 1 else 0
    team2 = st.selectbox("Team 2", options=team_options, index=default_team2_index)

if team1 == team2:
    st.warning("Team 1 and Team 2 are the same. Please select different teams.")

col3, col4 = st.columns(2)
with col3:
    toss_winner = st.selectbox("Toss Winner", options=[team1, team2], index=0)
with col4:
    toss_decision = st.selectbox("Toss Decision", options=toss_decision_options, index=0)

col5, col6 = st.columns(2)
with col5:
    team1_form_winrate_5 = st.number_input(
        "Team 1 Form Winrate (Last 5)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
with col6:
    team2_form_winrate_5 = st.number_input(
        "Team 2 Form Winrate (Last 5)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

col7, col8 = st.columns(2)
with col7:
    venue_chase_winrate_prior = st.number_input(
        "Venue Chase Winrate Prior", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
with col8:
    venue_score_prior = st.number_input(
        "Venue Score Prior", min_value=-1.0, max_value=1.0, value=0.0, step=0.01
    )

if st.button("Predict Winner", type="primary"):
    if team1 == team2:
        st.error("Please select two different teams.")
        st.stop()

    payload = {
        "Team1": team1,
        "Team2": team2,
        "Toss_Winner": toss_winner,
        "Toss_Decision": toss_decision,
        "team1_form_winrate_5": float(team1_form_winrate_5),
        "team2_form_winrate_5": float(team2_form_winrate_5),
        "venue_chase_winrate_prior": float(venue_chase_winrate_prior),
        "venue_score_prior": float(venue_score_prior),
    }

    input_row = pd.DataFrame([payload])[feature_columns]
    pred_encoded = model.predict(input_row)[0]
    pred_proba = model.predict_proba(input_row)[0]
    winner = label_encoder.inverse_transform([pred_encoded])[0]
    winner_prob = float(pred_proba[pred_encoded])

    st.success(f"Predicted Winner: {winner}")
    st.info(f"Winning Probability: {winner_prob * 100:.2f}%")
