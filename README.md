# Trilingual Sentiment LoRA

Multilingual sentiment analysis (English/Arabic/French) using XLM-R base with a LoRA adapter, bundled with a simple Streamlit app.

## Features
- Multilingual sentiment prediction: English, Arabic, Spanish.
- Streamlit UI with example inputs and confidence visualizations.
- Clean module layout: helpers, models, and API interface.
- Simple config loader for future parameterization.

## Project Structure
```
├── main.py                  # Streamlit app entry
└── src/
    ├── api/
    │   └── api.py           # Model loading and analyze_sentiment() function
    ├── helpers/
    │   ├── config.py        # Loads config.json with defaults
    │   └── config.json      # Optional overrides
    ├── models/
    │   └── pipeline.py      # Generic sentiment pipeline + rule-based fallback
    └── assets/              # UI assets (placeholder)
```

## Requirements
- Python 3.9+
- `pip install -r requirements.txt`
  - Includes `streamlit`, `transformers`, `torch`, `peft`, `plotly`.

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run main.py
   ```
3. Open the app at `http://localhost:8501`.

## Usage
- Enter text in the main input and click "Analyze".
- Use the example buttons (EN/AR/SP) to populate sample inputs.
- The app displays the predicted sentiment and confidence scores.

## Configuration
- `src/helpers/config.json` (optional). If present, values merge with defaults from `config.py`.
- Defaults include model name and label mapping; current Streamlit UI uses `src/api/api.py` which loads:
  - Base model: `FacebookAI/xlm-roberta-base`
  - Adapter: `osamanaguib/trilingual-sentiment-lora`

## API Interface
- `src/api/api.py` exposes:
  - `analyze_sentiment(text: str) -> dict`
    - Returns `{ input, sentiment, scores }` where `scores` are label:confidence.

## Troubleshooting
- Missing `streamlit` command:
  - Ensure `pip install -r requirements.txt` succeeded and environment is active.

- AttributeError: `module 'streamlit' has no attribute 'experimental_rerun'`:
  - Recent Streamlit versions replaced `st.experimental_rerun()` with `st.rerun()`.
  - Update code to use `st.rerun()` when you need to force a re-run after changing `st.session_state`.

- Slow first prediction:
  - Model and adapter weights load on first use. Subsequent predictions are faster.

## Notes
- If predictions look off, verify adapter weights are accessible and label mapping matches adapter training.
- Consider moving large model weights out of the repo or using Git LFS/DVC.
