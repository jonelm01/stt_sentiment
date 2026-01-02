import streamlit as st
import time
from stt_utils import *

st.set_page_config(page_title="Speech-to-Text + Sentiment", layout="wide")

# Session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "file_data" not in st.session_state:
    st.session_state.file_data = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "transcription_started" not in st.session_state:
    st.session_state.transcription_started = False  # flag to show containers

MAX_HISTORY = 50

# Load models
whisper_model = load_whisper()
sent_tokenizer, sent_model = load_sentiment()

# UI
st.header("Speech-to-Text + Sentiment")
st.caption("Manual transcription • Human labeling • CSV logging")
tabs = st.tabs(["Analyze", "History"])

with tabs[0]:
    left, right = st.columns([1.35, 1], gap="large")

    # Audio upload
    with left:
        files = st.file_uploader(
            "Upload one or more files",
            type=["mp3", "wav", "m4a"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if files:
            st.session_state.uploaded_files = files
        elif not st.session_state.uploaded_files:
            st.info("Upload at least one audio file")

    # Start / Reset
    start = st.button("Start transcription")
    reset = st.button("Reset")

    # Config panel
    with right:
        st.subheader("Configuration")
        st.write("Whisper:", FW_MODEL_ID, "| Device:", FW_DEVICE, "| Compute:", FW_COMPUTE)

    # Reset
    if reset:
        st.session_state.uploaded_files.clear()
        st.session_state.file_data.clear()
        st.session_state.transcription_started = False

    # Start transcription
    if start and st.session_state.uploaded_files:
        st.session_state.transcription_started = True

    # Show containers after transcription started
    if st.session_state.transcription_started:
        for uploaded in st.session_state.uploaded_files:
            run_id = stable_run_id(uploaded.getvalue())

            if run_id not in st.session_state.file_data:
                st.session_state.file_data[run_id] = {
                    "filename": uploaded.name,
                    "text": "",
                    "pred_label": "Neutral",
                    "probs": [0.0, 1.0, 0.0],
                    "confidence": 0.0,
                    "human_label": "Neutral",
                    "transcribed": False
                }

            data = st.session_state.file_data[run_id]

            # Form for each file (prevents rerun issues)
            with st.form(key=f"form_{run_id}"):
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        padding: 15px;
                        margin-bottom: 20px;
                        ">
                        <h4>{uploaded.name}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Transcription
                if not data["transcribed"]:
                    with st.spinner(f"Transcribing {uploaded.name} ..."):
                        audio = load_audio(uploaded.getvalue())
                        text = transcribe(audio, whisper_model)
                        pred_label, probs = analyze_sentiment(text, sent_tokenizer, sent_model)
                        conf = float(max(probs))

                        data.update({
                            "text": text,
                            "pred_label": pred_label,
                            "probs": probs,
                            "confidence": conf,
                            "human_label": pred_label,
                            "transcribed": True
                        })

                # Transcript
                st.text_area(f"Transcript_{run_id}", data["text"], height=260, label_visibility="collapsed")

                # Sentiment
                st.metric("Model result", data["pred_label"])
                st.caption(f"Confidence: {data['confidence']:.3f}")

                # Human labeling
                selected_label = st.radio(
                    "Human label",
                    LABELS_UI,
                    index=LABELS_UI.index(data["human_label"]),
                    horizontal=True,
                    key=f"human_{run_id}"
                )
                data["human_label"] = selected_label

                # Save 
                submitted = st.form_submit_button(f"Save {uploaded.name}")
                if submitted:
                    append_hitl_row({
                        "id": run_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "text": data["text"],
                        "pred_label": data["pred_label"].lower(),
                        "pred_conf": data["confidence"],
                        "p_neg": float(data["probs"][0]),
                        "p_neu": float(data["probs"][1]),
                        "p_pos": float(data["probs"][2]),
                        "human_label": data["human_label"].lower(),
                        "whisper_model": FW_MODEL_ID,
                    })
                    st.success(f"{uploaded.name} saved!")

                    # Update history
                    st.session_state.history.append({
                        "run_id": run_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "text": data["text"],
                        "pred_label": data["pred_label"],
                        "confidence": data["confidence"],
                        "p_neg": float(data["probs"][0]),
                        "p_neu": float(data["probs"][1]),
                        "p_pos": float(data["probs"][2]),
                    })
                    if len(st.session_state.history) > MAX_HISTORY:
                        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

# History tab
with tabs[1]:
    st.subheader("Run history")
    if not st.session_state.history:
        st.info("No runs yet")

    for i, r in enumerate(reversed(st.session_state.history)):
        with st.expander(f"{r['timestamp']} • Sentiment: {r['pred_label']} ({r['confidence']:.2f})"):
            st.text_area(
                "Transcript",
                r["text"],
                height=140,
                label_visibility="collapsed",
                key=f"hist_{i}_{r['run_id']}"
            )
            st.caption(f"Probabilities - Neg: {r['p_neg']:.3f}, Neu: {r['p_neu']:.3f}, Pos: {r['p_pos']:.3f}")
