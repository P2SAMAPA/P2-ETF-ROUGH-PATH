# Rough Path SDE Engine

Simulates fractional Brownian motion (Hurst = 0.3) and solves controlled rough differential equations using Davie's expansion. Models rough volatility without discretisation error. Predicts next‑day ETF returns.

- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Hurst exponent:** 0.3 (rough regime)
- **Davie expansion order:** 2
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
