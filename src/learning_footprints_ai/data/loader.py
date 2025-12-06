import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

class StockDatasetLoader:
    """
    Generates or loads a stock-like dataset and prepares
    normalized train/val/test splits as PyTorch tensors.
    """

    def __init__(self, num_samples: int = 2000, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)

    def generate_synthetic_data(self):
        """
        Create a synthetic stock dataset with 6 features and a binary label.
        """

        # Features (stock-like signals)
        price_change = np.random.normal(0, 1, self.num_samples)
        volume_change = np.random.normal(0, 1.5, self.num_samples)
        volatility = np.random.uniform(0.1, 2.0, self.num_samples)
        ma_cross = np.random.normal(0, 1, self.num_samples)         # moving average crossover
        momentum = np.random.normal(0, 1, self.num_samples)
        sentiment = np.random.normal(0, 1, self.num_samples)

        # Combine features
        X = np.vstack([
            price_change,
            volume_change,
            volatility,
            ma_cross,
            momentum,
            sentiment
        ]).T

        # Hidden "true" rule for labels
        logits = (
            0.8 * price_change
            + 0.5 * volume_change
            + 0.3 * sentiment
            + 0.2 * momentum
            - 0.4 * volatility
        )

        # Convert logits → probability → binary class
        y = (logits > np.median(logits)).astype(int)

        return X, y

    def prepare_data(self):
        """
        Generates data, normalizes features, and returns PyTorch tensors.
        """

        # 1. Create synthetic dataset
        X, y = self.generate_synthetic_data()

        # 2. Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. Split into train/val/test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=0.30, random_state=self.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.seed
        )

        # 4. Convert to PyTorch tensors
        train_data = torch.tensor(X_train, dtype=torch.float32)
        train_labels = torch.tensor(y_train, dtype=torch.long)

        val_data = torch.tensor(X_val, dtype=torch.float32)
        val_labels = torch.tensor(y_val, dtype=torch.long)

        test_data = torch.tensor(X_test, dtype=torch.float32)
        test_labels = torch.tensor(y_test, dtype=torch.long)

        return {
            "train": (train_data, train_labels),
            "val": (val_data, val_labels),
            "test": (test_data, test_labels),
            "input_dim": X.shape[1]
        }
