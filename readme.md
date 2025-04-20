Here’s a clean and structured **README.md** file for your time series prediction project, covering setup instructions, usage, and Flask API endpoints.

---

# 🧠 Disease Time Series Prediction Model
```markdown
This project implements a time series prediction system to forecast daily new cases for various diseases (like COVID, Malaria, etc.). It supports multiple ML and DL models including SVR, Random Forest, Linear Regression, and LSTM. The application is wrapped in a Flask API with endpoints for training/visualization and prediction.

```

## 📁 Project Structure
```
.
├── data/
│   └── covid/
│       └── covid.csv
│   └── malaria/
│       └── malaria.csv
├── image/
│   └── predicted.png
│   └── mean.png
├── app.py
├── prediction_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 🐍 Create & Activate Conda Environment

```bash
conda create --name tf_env python=3.10
conda activate tf_env
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Flask API Usage

### ▶️ 1. Start the Flask Server

```bash
python app.py
```

This will launch a local Flask API server.

---

## 🧪 API Endpoints

### 🔧 `/train`

**Method**: `POST`  
**Description**: Trains the model on the disease dataset, generates visualizations, and saves plots.

**Request Body (JSON)**:
```json
{
  "disease": "covid",
  "model_name": "random_forests"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model trained and plots saved."
}
```

---

### 🤖 `/predict`

**Method**: `POST`  
**Description**: Predicts the next day's new cases using the last `n_lags` values.

**Request Body (JSON)**:
```json
{
  "disease": "covid",
  "model_name": "random_forests",
  "recent_cases": [100, 120, 130, 140, 150, 160, 170]
}
```

> Note: `recent_cases` must be exactly `n_lags` long (default is 7).

**Response**:
```json
{
  "predicted_next_case": 183.27
}
```

---

## 📊 Output Charts

After training, the following charts are saved under the `/image` folder:
- `predicted.png`: Line chart comparing actual vs predicted new cases
- `mean.png`, `min.png`, `max.png`, `median.png`: Aggregated trend visualizations

---

## 💻 Supported Models

- `svr_linear`
- `svr_poly`
- `svr_rbf`
- `linear_regression`
- `random_forests`
- `knn`
- `elastic_net`
- `dt` (Decision Tree)
- `lstm_model` (Deep Learning)

---
