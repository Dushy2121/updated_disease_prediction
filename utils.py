import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import os
from tqdm.keras import TqdmCallback
from tqdm import tqdm


class PredictionModel:

    def __init__(self, disease: str, model_name: str, test_size: float = 0.2,
                 random_state: int = 42, n_lags: int = 7):
        self.disease = disease.lower()
        self.model_name = model_name.lower()
        self.test_size = test_size
        self.random_state = random_state
        self.n_lags = n_lags

        self.model = None
        self.scaler = None
        self.date_col = None
        self.target_col = 'New_cases'

        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.dates_test = None

        self._load_data()

    def _load_data(self):
        filepath = f"data/{self.disease}/{self.disease}.csv"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")

        df = pd.read_csv(filepath)

        if self.disease == 'malaria':
            self.date_col = 'Date_reported'
        else:
            self.date_col = 'date'

        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df.sort_values(self.date_col, inplace=True)
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
        df.dropna(subset=[self.target_col], inplace=True)

        self.dates = df[self.date_col].values
        self.y = df[self.target_col].values

    def _prepare_data(self): # TODO ask gpt what this function does
        split_idx = int(len(self.y) * (1 - self.test_size))
        y_train, y_test = self.y[:split_idx], self.y[split_idx:]
        dates_train, dates_test = self.dates[:split_idx], self.dates[split_idx:]

        if self.model_name == 'lstm_model':
            return y_train, y_test, dates_train, dates_test

        data = pd.DataFrame({'y': np.concatenate([y_train, y_test])})
        for lag in range(1, self.n_lags + 1):
            data[f'lag_{lag}'] = data['y'].shift(lag)

        data.dropna(inplace=True)
        dates_aligned = self.dates[self.n_lags:]

        values = data['y'].values
        split_aligned = split_idx - self.n_lags

        X = data[[f'lag_{i}' for i in range(1, self.n_lags + 1)]].values
        self.X_train, self.X_test = X[:split_aligned], X[split_aligned:]
        self.y_train, self.y_test = values[:split_aligned], values[split_aligned:]
        self.dates_test = dates_aligned[split_aligned:]

        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def _initialize_model(self):
        models = {
            'svr_linear': SVR(kernel='linear'),
            'svr_poly': SVR(kernel='poly'),
            'svr_rbf': SVR(kernel='rbf'),
            'linear_regression': LinearRegression(),
            'random_forests': RandomForestRegressor(random_state=self.random_state),
            'knn': KNeighborsRegressor(),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'dt': DecisionTreeRegressor(random_state=self.random_state)
        }
        return models.get(self.model_name)

    def train(self):
        if self.model_name == 'lstm_model':
            y_train, y_test, _, self.dates_test = self._prepare_data()
            generator = TimeseriesGenerator(y_train, y_train, length=self.n_lags, batch_size=1)

            self.model = Sequential([
                LSTM(50, activation='relu', input_shape=(self.n_lags, 1)),
                Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(generator, epochs=20, verbose=0, callbacks=[TqdmCallback(verbose=0)])

            preds, history = [], list(y_train)
            for _ in tqdm(range(len(y_test)), desc="LSTM Forecasting Progress"):
                seq = np.array(history[-self.n_lags:]).reshape((1, self.n_lags, 1))
                yhat = self.model.predict(seq, verbose=0)[0, 0]
                preds.append(yhat)
                history.append(yhat)

            self.y_test = y_test
            self.y_pred = np.array(preds)

        else:
            self._prepare_data()
            self.model = self._initialize_model()
            if self.model is None:
                raise ValueError(f"Unsupported model: {self.model_name}")
            print(f"Training {self.model_name} model...")
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)


    def plot_results(self, save_dir="image"):
        os.makedirs(save_dir, exist_ok=True)

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

        df_plot = pd.DataFrame({
            'date': self.dates_test,
            'actual': self.y_test,
            'predicted': self.y_pred
        })
        agg = df_plot.groupby('date').agg(
            actual_min=('actual', 'min'),
            actual_max=('actual', 'max'),
            actual_mean=('actual', 'mean'),
            actual_median=('actual', 'median'),
            pred_min=('predicted', 'min'),
            pred_max=('predicted', 'max'),
            pred_mean=('predicted', 'mean'),
            pred_median=('predicted', 'median')
        ).reset_index()

        stats = ['min', 'max', 'mean', 'median']
        for stat in stats:
            plt.figure(figsize=(10, 5))
            plt.plot(agg['date'], agg[f'actual_{stat}'], label='Actual', marker='o')
            plt.plot(agg['date'], agg[f'pred_{stat}'], label='Predicted', linestyle='--', marker='x')
            plt.xticks(rotation=45)
            plt.title(f"{self.disease.capitalize()} - {self.model_name} {stat.capitalize()}")
            plt.xlabel('Date')
            plt.ylabel('New Cases')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{stat}.png")

    def predict_next(self, recent_cases):
        """
        Predict the next day's value given recent real values.
        """
        if len(recent_cases) != self.n_lags:
            raise ValueError(f"Expected {self.n_lags} recent cases, got {len(recent_cases)}")

        if self.model_name == 'lstm_model':
            input_seq = np.array(recent_cases).reshape((1, self.n_lags, 1))
            return self.model.predict(input_seq, verbose=0)[0, 0]
        else:
            input_array = np.array(recent_cases).reshape(1, -1)
            if self.scaler:
                input_array = self.scaler.transform(input_array)
            return self.model.predict(input_array)[0]
    
    def get_model_list():
        return [
            'svr_linear','svr_poly','svr_rbf','linear_regression','random_forests','knn','elastic_net','Decision Tree Regressor'
        ]


model = PredictionModel(disease='covid', model_name='knn')
model.train()
model.plot_results()

# Predict new case
recent_cases= [100, 120, 130, 140, 150, 160, 170]
next_case = model.predict_next(recent_cases)
print("Predicted Next Case:", next_case)
