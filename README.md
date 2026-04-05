# Detection of Climate Change Signals in Indian Temperature Records
### Using Probability Theory and Moving Average Processes for Anomaly Identification and Trend Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ARMA](https://img.shields.io/badge/Model-ARMA-green)

---

## Overview

This project analyses 70+ years of daily temperature data across 5 major Indian cities — **Delhi, Mumbai, Bengaluru, Chennai and Kolkata** — to detect climate change signals and model temperature patterns using the ARMA (AutoRegressive Moving Average) model.

---

## Cities and Dataset

| City | Region | Date Range |
|------|--------|------------|
| Delhi | North India | 1951 – 2024 |
| Mumbai | West Coast | 1951 – 2024 |
| Bengaluru | South India | 1951 – 2024 |
| Chennai | East Coast | 1951 – 2024 |
| Kolkata | East India | 1951 – 2024 |

Each CSV file contains: `Date`, `Temp Max`, `Temp Min`, `Rain`

---

## Project Structure

```
├── app.py               # Streamlit web app
├── ARMA_MODEL.ipynb     # Jupyter notebook (step by step analysis)
├── delhi.csv
├── mumbai.csv
├── bengaluru.csv
├── chennai.csv
├── kolkata.csv
└── README.md
```

---

## Methodology

1. **Data Loading and Cleaning** — Computed `Temp Mean = (Temp Max + Temp Min) / 2`, removed outliers above 45°C
2. **Visualisation** — Plotted 70 years of daily temperature for each city
3. **Stationarity Test** — ADF (Augmented Dickey-Fuller) test confirmed all 5 cities are stationary
4. **Baseline Model** — ARMA(1,1) fitted for all cities as a starting point
5. **MAE Evaluation** — Train on 1951–2020, test on 2021–2024
6. **Grid Search** — MAE-based grid search over p, q in range 0–3 to find the best model per city
7. **Final Comparison** — Baseline vs best model MAE comparison

---

## Results

| City | Best Model | MAE (°C) |
|------|-----------|----------|
| Delhi | ARMA(3,3) | 5.82 |
| Mumbai | ARMA(0,0) | 2.55 |
| Bengaluru | ARMA(1,0) | 1.78 |
| Chennai | ARMA(3,0) | 2.25 |
| Kolkata | ARMA(3,3) | 3.57 |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Gauri210/arma-temperature-analysis.git
cd arma-temperature-analysis
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy matplotlib statsmodels scikit-learn
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

---

## Limitations

- ARMA cannot fully capture strong seasonal patterns in daily temperature data
- Forecast converges to the mean rather than following the seasonal cycle
- A seasonal model like **SARIMA** would perform significantly better for long-term forecasting

---

## References

- Chatterjee, J. (2026). *Temperature trends in India*. Data For India. https://www.dataforindia.com/temperature-trends-in-india/
- Medhi, J. *Stochastic Processes*, 5th Edition, New Age International, 2017.
