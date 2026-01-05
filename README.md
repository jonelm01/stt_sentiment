# SupportPulse

**SupportPulse** is a speech-to-text and sentiment analysis application designed for **customer support and e-commerce support teams**. It analyzes recorded support calls to determine the **overall sentiment of customer interactions**, helping teams identify potentially negative experiences for triage, escalation, and quality assurance.

The application focuses on **transcript-based sentiment analysis**, with optional **human-in-the-loop review** to improve reliability and data quality over time.

---

## Problem Statement

Customer support calls contain valuable signals about customer satisfaction and frustration, but extracting those signals at scale is challenging.  
Support teams often rely on:

- manual call reviews,
- post-call surveys (e.g., CSAT),
- or reactive escalation after complaints surface.

These approaches are time-consuming and typically delay insight until well after the support interaction has concluded.

---

## Solution

SupportPulse provides a practical tool for converting customer support calls into **actionable sentiment insights**:

1. Transcribes recorded support calls into text.
2. Analyzes the transcript using a **sentiment classifier fine-tuned on customer support conversations**.
3. Produces an overall sentiment classification (Negative / Neutral / Positive) with confidence scores.
4. Allows a human reviewer to **confirm or override the model’s prediction**.
5. Stores validated sentiment labels for future analysis and model improvement.

The system is designed to support human decision-making rather than automate it fully.

---

## Key Features

### Speech-to-Text Transcription
- Converts recorded customer support calls into text using a Whisper-based transcription model.

### Domain-Specific Sentiment Analysis
- Uses a RoBERTa-based sentiment classifier fine-tuned on e-commerce customer support data.
- Outputs an **overall sentiment classification** for each conversation.

### Human-in-the-Loop Review
- After sentiment prediction, the user can:
  - confirm the model’s sentiment, or
  - select a different sentiment label.
- Final, user-confirmed labels are saved to a **CSV file**, enabling:
  - auditability,
  - dataset expansion,
  - and future retraining.

### Batch Upload Support
- Multiple audio files can be uploaded at once.
- Files are processed **one at a time** after loading, ensuring:
  - controlled execution,
  - clear per-file review,
  - and accurate human validation.

### Clear, Support-Oriented UI
- Displays:
  - dominant sentiment,
  - confidence scores,
  - full call transcript.
- Designed for fast review and operational clarity.

### Session History
- Analyzed calls are stored during the session for comparison and review.

---

## How It Works

1. **Audio Upload**  
   The user uploads one or more recorded customer support calls (`.mp3`, `.wav`, `.m4a`).

2. **Transcription**  
   Each audio file is transcribed into text.

3. **Sentiment Classification**  
   The transcript is passed to a fine-tuned sentiment model that predicts the **overall sentiment of the conversation**.

4. **Human Review**  
   The user reviews the predicted sentiment and can confirm or override it.

5. **Persistence**  
   The final sentiment decision is saved to a CSV file along with the transcript and metadata.

---

## Scope and Limitations

### What the Project Does
- Analyzes **text-based sentiment** derived from speech transcripts.
- Supports human validation and correction.
- Enables batch ingestion with controlled, per-item processing.

### What the Project Does Not Do (Currently)
- Analyze vocal tone, pitch, or prosody.
- Perform acoustic emotion or affect detection.
- Infer intent beyond textual sentiment.

---

## Human-in-the-Loop Design

A key design goal of SupportPulse is **trust and adaptability**.

Rather than treating model predictions as final:
- users can validate or override sentiment labels,
- corrected labels are persisted,
- and the system becomes a tool for **iterative dataset improvement**.

This design supports real-world support workflows where:
- sentiment can be ambiguous,
- context matters,
- and human judgment remains essential.

---

## Tech Stack

- **Frontend**: Streamlit  
- **Speech Recognition**: Whisper  
- **Sentiment Analysis**: RoBERTa (TensorFlow)  
- **Model Training**: TensorFlow + Hugging Face Transformers  
- **Data Handling**: Hugging Face Datasets  
- **Persistence**: CSV-based labeling store  
- **Language**: Python  

---

## Intended Use Cases

- Support call triage and escalation
- Quality assurance review prioritization
- Agent coaching and feedback
- CX analysis alongside survey metrics
- Building labeled datasets for future NLP improvements

---

## Future Enhancements

Planned or potential extensions include:
- sentiment trend dashboards,
- integration with ticketing systems,
- agent-level analytics,
- multilingual support,
- **tone and acoustic emotion detection** as a future enhancement layered on top of transcript-based sentiment.

---

## Summary

SupportPulse demonstrates how **speech recognition, domain-specific NLP, and human-in-the-loop validation** can be combined to create a practical tool for customer support teams. By focusing on transcript-based sentiment analysis and explicit human review, the project emphasizes reliability, interpretability, and real-world operational workflows over fully automated decision-making.
