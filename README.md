
# House Price Predictor 🏠💰

An end-to-end machine learning project that predicts house prices using an optimized XGBoost regression pipeline and features an interactive web dashboard built with Streamlit.

---

## 🚀 Features
* **Production-Grade Pipeline:** Features automated numerical/categorical preprocessing using `scikit-learn` pipelines, target transformation (`np.log1p`), and robust cross-validation.
* **Interactive Dashboard:** A clean user interface allowing real-time parameter tuning and immediate price prediction.
* **Lightweight Environment:** Built and managed efficiently using modern Python packaging tools (`uv`).

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Modeling & Pipelines:** `XGBoost`, `Scikit-learn`, `NumPy`, `Pandas`
* **Web Framework:** `Streamlit`
* **Environment Management:** `uv` / `pip`

---

## 📊 Model Training & Development
The core machine learning workflow is documented inside the `XG_Boost.ipynb` notebook. The development process includes:
1. **Data Preprocessing:** Handled via custom ColumnTransformers targeting data types dynamically.
2. **Target Engineering:** Log-transforming the `SalePrice` to stabilize variance and minimize prediction error metrics.
3. **Hyperparameter Tuning:** Optimized using `GridSearchCV` to prevent overfitting while boosting regression accuracy.

---

## 📦 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Akshat-Sharma2005/House-price-predictor.git](https://github.com/Akshat-Sharma2005/House-price-predictor.git)
cd House-price-predictor

```

### 2. Install Dependencies

If you use **uv**:

```bash
uv sync

```

Or using standard **pip**:

```bash
pip install -r requirements.txt

```

### 3. Run the Dashboard

```bash
streamlit run app.py

```
