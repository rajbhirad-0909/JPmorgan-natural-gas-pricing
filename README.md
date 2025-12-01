# Natural Gas Price Analysis & Forecasting (2020–2024)
### JP Morgan — Data Science & Markets Project  
**Author:** Raj Bhirad  

---

## 📌 Project Overview
This project analyzes monthly **Natural Gas Prices (2020–2024)** and builds a **forecasting model** using:

- Time-series trend analysis  
- Seasonal decomposition  
- Linear Regression  
- Month-based seasonality adjustments  

The final model can estimate future natural gas prices up to **12 months ahead** with a trend + seasonality approach.

---

## 📊 Dataset
- **File:** `Nat_Gas.csv`
- **Columns:**
  - `Dates` — Monthly time stamps  
  - `Prices` — Natural gas spot prices  
- Dataset cleaned and parsed using `pandas`.

---

## 🔧 Techniques Used
### **1. Data Preprocessing**
- Date parsing with `pandas.to_datetime`
- Handling missing values
- Sorting and indexing by date  
- Monthly numeric time index for modeling

### **2. Visualizations**
Four key charts generated using `matplotlib`:

1. **Raw Price Trend (2020–2024)**  
2. **Monthly Seasonality Pattern**  
3. **Linear Regression Trend Line**  
4. **12-Month Forecast (Trend + Seasonality)**  

All charts are saved automatically as PNG.

### **3. Model**
- Trend extracted using **Linear Regression**
- Seasonality learned from **monthly averages**
- Combined model:

- Forecast horizon: **12 months** beyond dataset

---

## 📁 Project Structure
│
├── Nat_Gas.csv
├── charts/
│ ├── chart_1_raw_prices.png
│ ├── chart_2_seasonality.png
│ ├── chart_3_trend_line.png
│ └── chart_4_forecast.png
│
└── gas_analysis.ipynb # Jupyter notebook with full analysis

---

## 🧠 Model Function

The project includes a reusable function:

```python
estimate_price("2025-02-15")
🚀 Key Results

Natural gas prices show a mild upward trend over 2020–2024.

Seasonality is present: certain months consistently higher/lower.

Combined model provides stable forward projections.

🛠 Technologies Used

Python 3.10+

pandas

NumPy

Matplotlib

scikit-learn

