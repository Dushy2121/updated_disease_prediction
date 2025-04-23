import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os
from tqdm.keras import TqdmCallback
from tqdm import tqdm


class PredictionModel:
    @staticmethod
    def get_model_list():
        return [
            'svr_linear', 'svr_poly', 'svr_rbf',
            'linear_regression', 'random_forests',
            'knn', 'elastic_net', 'decision_tree',
            'lstm_model'
        ]

    def __init__(
        self,
        disease: str,
        model_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        n_lags: int = 7,
        log_transform: bool = True,
        use_grid_search: bool = False,
        svr_degree: int = 3,
        dt_max_depth: int = 5,
        dt_min_samples_leaf: int = 5
    ):
        self.disease = disease.lower()
        self.model_name = model_name.lower()
        self.test_size = test_size
        self.random_state = random_state
        self.n_lags = n_lags
        self.log_transform = log_transform
        self.use_grid_search = use_grid_search
        self.svr_degree = svr_degree
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_leaf = dt_min_samples_leaf

        # For SVR models, work on raw scale only
        if self.model_name.startswith('svr_'):
            self.log_transform = False

        self.model = None
        self.scaler = StandardScaler()
        self.date_col = None
        self.target_col = 'New_cases'

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.dates_test = None

        self._load_data()
        self._prepare_data()

    def _load_data(self):
        filepath = f"data/{self.disease}/{self.disease}.csv"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")

        df = pd.read_csv(filepath)
        self.date_col = 'Date_reported' if self.disease == 'malaria' else 'date'
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df.sort_values(self.date_col, inplace=True)
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
        df.dropna(subset=[self.target_col], inplace=True)

        self.dates = df[self.date_col].values
        y = df[self.target_col].values.reshape(-1, 1)
        self.y = np.log1p(y) if self.log_transform else y

    def _prepare_data(self):
        n_samples = len(self.y)
        if n_samples <= self.n_lags + 1:
            raise ValueError(f"Not enough data: got {n_samples}, need > n_lags ({self.n_lags})")

        split_idx = int(n_samples * (1 - self.test_size))
        if split_idx <= self.n_lags or split_idx >= n_samples:
            raise ValueError(f"Empty train/test: test_size={self.test_size}, n_lags={self.n_lags}")

        y_train_full = self.y[:split_idx]
        y_test_full = self.y[split_idx:]
        dates_full = self.dates

        if self.model_name == 'lstm_model':
            self.generator = TimeseriesGenerator(
                y_train_full, y_train_full,
                length=self.n_lags, batch_size=1
            )
            self.dates_test = dates_full[split_idx:]
            self.y_train = y_train_full.flatten()
            self.y_test = y_test_full.flatten()
            return

        df_lag = pd.DataFrame({'y': self.y.flatten()})
        for lag in range(1, self.n_lags + 1):
            df_lag[f'lag_{lag}'] = df_lag['y'].shift(lag)
        df_lag.dropna(inplace=True)

        X_all = df_lag[[f'lag_{i}' for i in range(1, self.n_lags + 1)]].values
        y_all = df_lag['y'].values.reshape(-1, 1)
        dates_lagged = dates_full[self.n_lags:]

        split_lag = split_idx - self.n_lags
        if split_lag < 1 or split_lag >= len(X_all):
            raise ValueError(f"After lagging, split {split_lag} yields empty set. Adjust test_size or n_lags.")

        self.X_train = X_all[:split_lag]
        self.X_test = X_all[split_lag:]
        self.y_train = y_all[:split_lag].flatten()
        self.y_test = y_all[split_lag:].flatten()
        self.dates_test = dates_lagged[split_lag:]

        self.X_train = self.scaler.fit_transform(self.X_train)
        if len(self.X_test) > 0:
            self.X_test = self.scaler.transform(self.X_test)

    def _initialize_model(self):
        models = {
            'svr_linear': SVR(kernel='linear'),
            'svr_poly': SVR(kernel='poly', degree=self.svr_degree),
            'svr_rbf': SVR(kernel='rbf'),
            'linear_regression': LinearRegression(),
            'random_forests': RandomForestRegressor(
                random_state=self.random_state, n_estimators=100
            ),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'elastic_net': ElasticNet(
                random_state=self.random_state, alpha=1.0, l1_ratio=0.5
            ),
            'decision_tree': DecisionTreeRegressor(
                random_state=self.random_state,
                max_depth=self.dt_max_depth,
                min_samples_leaf=self.dt_min_samples_leaf
            )
        }
        return models.get(self.model_name)

    def train(self):
        if self.model_name == 'lstm_model':
            self.model = Sequential([
                LSTM(50, activation='relu', input_shape=(self.n_lags, 1)),
                Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(
                self.generator, epochs=20,
                verbose=0, callbacks=[TqdmCallback(verbose=0)]
            )

            history = list(self.y_train)
            preds = []
            for _ in tqdm(range(len(self.y_test)), desc="Forecasting"):
                seq = np.array(history[-self.n_lags:]).reshape((1, self.n_lags, 1))
                yhat = self.model.predict(seq, verbose=0)[0, 0]
                preds.append(yhat)
                history.append(yhat)

            self.y_pred = np.array(preds)
        else:
            model = self._initialize_model()
            if model is None:
                raise ValueError(f"Unsupported model: {self.model_name}")

            if self.use_grid_search and self.model_name.startswith('svr_'):
                param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1]}
                self.model = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            elif self.use_grid_search and self.model_name == 'random_forests':
                self.model = GridSearchCV(
                    model, {'n_estimators': [50, 100]}, cv=3,
                    scoring='neg_mean_squared_error'
                )
            elif self.use_grid_search and self.model_name == 'decision_tree':
                self.model = GridSearchCV(
                    model,
                    {'max_depth': [3, 5, 7], 'min_samples_leaf': [1, 5, 10]},
                    cv=3, scoring='neg_mean_squared_error'
                )
            else:
                self.model = model

            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

            # Bound predictions within training range
            lower, upper = np.min(self.y_train), np.max(self.y_train)
            self.y_pred = np.clip(self.y_pred, lower, upper)

            print(f"Raw {self.model_name} outputs (bounded):", self.y_pred[:5])

        if self.log_transform and len(self.y_pred) > 0:
            self.y_pred = np.expm1(self.y_pred)
            self.y_test = np.expm1(self.y_test)

    def evaluate(self):
        if len(self.y_test) == 0:
            raise ValueError("No test samples available.")
        return {
            'mse': mean_squared_error(self.y_test, self.y_pred),
            'mae': mean_absolute_error(self.y_test, self.y_pred)
        }

    def plot_results(self, save_dir="image"):
        os.makedirs(save_dir, exist_ok=True)
        if len(self.y_test) == 0:
            print("No test data to plot.")
            return
        plt.figure(figsize=(12, 5))
        plt.plot(self.dates_test, self.y_test, label='Actual', marker='o')
        plt.plot(self.dates_test, self.y_pred, label='Predicted', linestyle='--', marker='x')
        plt.xticks(rotation=45)
        plt.title(f"{self.disease.capitalize()} - {self.model_name} Predictions")
        plt.xlabel('Date')
        plt.ylabel('New Cases')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/predicted.png")

    def predict_next(self, recent_cases):
        if len(recent_cases) != self.n_lags:
            raise ValueError(f"Expected {self.n_lags} values, got {len(recent_cases)}")

        seq = np.array(recent_cases).astype(float).reshape(1, self.n_lags)
        if self.log_transform:
            seq = np.log1p(seq)
        seq = self.scaler.transform(seq)
        if self.model_name == 'lstm_model':
            seq = seq.reshape((1, self.n_lags, 1))

        pred = self.model.predict(seq)
        if self.log_transform:
            pred = np.expm1(pred)
        lower, upper = np.min(self.y_train if not self.log_transform else np.expm1(self.y_train)),np.max(self.y_train if not self.log_transform else np.expm1(self.y_train))
        pred = np.clip(pred, lower, upper)
        return float(pred[0])