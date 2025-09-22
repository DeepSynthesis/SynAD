from pathlib import Path
from hyperopt import hp
from loguru import logger
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchbnn as bnn
import torchbnn.functional as bnnF
from matplotlib import pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)
model_path = Path(__file__).parent / Path("../logs")


class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()

        if not (type(hidden_size) == list and len(hidden_size) == num_hidden_layers):
            hidden_size = [hidden_size] * num_hidden_layers

        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        self.layers.append(nn.BatchNorm1d(hidden_size[0]))
        self.layers.append(nn.LeakyReLU())

        for i in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            self.layers.append(nn.BatchNorm1d(hidden_size[i + 1]))
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.LeakyReLU())

        self.output = nn.Linear(hidden_size[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class NeuralNetworkRegressor:
    """Simple neural network construction"""

    def __init__(self, hidden_size, num_hidden_layers, learning_rate=0.001, epochs=100, dropout=1):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        X_train, y_train = X_train.astype(float), y_train.astype(float)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available and move model to GPU if it is
        if device == "cpu":
            logger.warning("Running on CPU")
        if X_test is None or y_test is None:
            logger.warning("No test data provided, not calculating R2 score")
        self.model = MyNeuralNetwork(X_train.shape[1], self.hidden_size, self.num_hidden_layers, self.dropout).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.75, patience=100)

        # Convert data to torch tensors and move to GPU if available
        X_train_tensor = torch.from_numpy(X_train).float().to(device)
        y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(device)

        # Training loop with tqdm for progress display
        progress_bar = tqdm(range(self.epochs), desc="Training", unit="epoch")
        r2_train, r2_eval = -np.inf, -np.inf
        for epoch in progress_bar:
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)  # automated lr adjustment
            if epoch % 10 == 0 and X_test is not None and y_test is not None:
                y_pred_train = self.predict(X_train).reshape(-1)
                y_pred_test = self.predict(X_test).reshape(-1)
                r2_train = round(float(r2_score(y_train, y_pred_train)), 3)
                r2_eval = round(float(r2_score(y_test, y_pred_test)), 3)

            # Update progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item(), r2_train=r2_train, r2_val=r2_eval, lr=self.optimizer.param_groups[0]["lr"])

        return self

    def predict(self, X_test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test = torch.from_numpy(X_test).float().to(device)
        with torch.no_grad():
            outputs = self.model(X_test)
        if device == "cpu":
            return outputs.numpy()
        else:
            return outputs.cpu().numpy()


class MyBayesianNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()

        if not (type(hidden_size) == list and len(hidden_size) == num_hidden_layers):
            hidden_size = [hidden_size] * num_hidden_layers

        self.layers.append(bnn.BayesLinear(in_features=input_size, out_features=hidden_size[0], prior_mu=0.5, prior_sigma=0.1))
        self.layers.append(nn.BatchNorm1d(hidden_size[0]))
        self.layers.append(nn.LeakyReLU())

        for i in range(num_hidden_layers - 1):
            self.layers.append(bnn.BayesLinear(in_features=hidden_size[i], out_features=hidden_size[i + 1], prior_mu=0, prior_sigma=0.1))
            self.layers.append(nn.BatchNorm1d(hidden_size[i + 1]))
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.LeakyReLU())

        self.output = bnn.BayesLinear(in_features=hidden_size[-1], out_features=1, prior_mu=0, prior_sigma=0.1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class BayesianNeuralNetworkRegressor:
    """A Bayesian neural network regressor."""

    def __init__(self, hidden_size, num_hidden_layers, learning_rate=1e-3, epochs=100, dropout=0.1, batch_size=2048):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.batch_size = batch_size

        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.model = None
        self.criterion = nn.MSELoss()

    def standardize_X_train(self, X_train):
        X_train = self.X_scaler.fit_transform(X_train)
        return X_train

    def standardize_X_test(self, X_test):
        X_test = self.X_scaler.transform(X_test)
        return X_test

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        X_train_scaled = self.X_scaler.fit_transform(X_train)
        y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        X_test_scaled = self.X_scaler.transform(X_test) if X_test is not None else None
        self.model = MyBayesianNeuralNetwork(X_train_scaled.shape[-1], self.hidden_size, self.num_hidden_layers, self.dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        lr_patience = 1000
        lr_factor = 0.8
        lr_counter = 0
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)

        X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1).to(device)

        train_data = TensorDataset(X_tensor.to(device), y_tensor.to(device))
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        progress_bar = tqdm(range(self.epochs), desc="Training", unit="epoch")
        r2_train, r2_eval = -np.inf, -np.inf
        current_lr = self.learning_rate
        for epoch in progress_bar:
            self.model.train()
            self.optimizer.zero_grad()
            for batch_X, batch_y in train_loader:
                outputs = self.model(batch_X)
                mse_loss = self.criterion(outputs, batch_y)
                kl_loss = bnnF.bayesian_kl_loss(self.model, reduction="mean")
                total_loss = mse_loss + kl_loss  # loss calculation
                total_loss.backward()
                self.optimizer.step()

            if epoch % 50 == 0:
                y_pred_train = self.predict(X_train_scaled).reshape(-1)
                r2_train = round(float(r2_score(y_train, y_pred_train)), 3)
                if X_test is not None and y_test is not None:
                    y_pred_test = self.predict(X_test_scaled).reshape(-1)
                    r2_eval = round(float(r2_score(y_test, y_pred_test)), 3)
                progress_bar.set_postfix(loss=total_loss.item(), r2_train=r2_train, r2_val=r2_eval, lr=self.learning_rate)

            if r2_train >= 0.95:  # early break
                break

            lr_counter += 1
            if lr_counter >= lr_patience:
                current_lr = self.optimizer.param_groups[0]["lr"] * lr_factor
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr
                lr_counter = 0

            progress_bar.set_postfix(loss=total_loss.item(), r2_train=r2_train, r2_val=r2_eval, lr=current_lr)
        return self

    def data_X_scale(self, X):
        return self.X_scaler.transform(X)

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(next(self.model.parameters()).device)
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy()
        pred = self.y_scaler.inverse_transform(pred)
        pred = np.clip(pred, 0, 100)
        return pred.flatten()

    def predict_with_uncertainty(self, x, num_samples=1000):
        """Monto Carlo's method for uncertainty estimation"""
        predictions = []
        for _ in range(num_samples):
            pred = self.predict(x).reshape(-1, 1)
            pred = np.clip(pred, 0, 100)
            predictions.append(pred)
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)  #  uncertainty part
        return mean, std


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, dropout):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        # GCN layers
        self.gc1 = GCNLayer(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.gc2 = GCNLayer(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x


class GCNRegressor:
    def __init__(self, hidden_dim, output_dim, num_hidden_layers, dropout):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.model = None

    def fit(self, X_global_data, X_atom_data, X_bond_data, y_data, epochs):
        self.model = GCN(X_atom_data.shape[1], self.hidden_dim, self.output_dim, self.num_hidden_layers, self.dropout)
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_global, X_atom, X_bond, y in zip(X_global_data, X_atom_data, X_bond_data, y_data):
                optimizer.zero_grad()
                output = self.model(X_atom, X_bond)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("Epoch {}, Loss: {:.4f}".format(epoch + 1, total_loss / len(y_data)))

    def predict(self, X_global_data, X_atom_data, X_bond_data):
        if self.model is None:
            raise ValueError("Model not trained. Please run fit method first.")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_global, X_atom, X_bond in zip(X_global_data, X_atom_data, X_bond_data):
                output = self.model(X_atom, X_bond)
                predictions.extend(output.cpu().numpy())
        return predictions


class GaussianNN(nn.Module):
    def __init__(self):
        super(GaussianNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, 1)
        self.fc_std = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.fc_std(x))
        return mean, std

    def negative_log_likelihood(y_pred, y_true, std):
        mean, std = y_pred
        log_likelihood = torch.log(std) + 0.5 * ((y_true - mean) / std) ** 2
        return torch.mean(log_likelihood)


NN_MODELS = {"NN": NeuralNetworkRegressor, "GCN": GCNRegressor, "BNN": BayesianNeuralNetworkRegressor}

NN_MODEL_HYPER_PARAM = {
    "NN_hyper_parameters": {
        "hidden_size": [1000, 2000, 500, 500],
        "num_hidden_layers": 4,
        "learning_rate": 0.01,
        "epochs": 5000,
        "dropout": 0.2,
    },
    "GCN_hyper_parameters": {
        "hidden_dim": 100,
        "output_dim": 1,
        "num_hidden_layers": 2,
        "dropout": 0.0,
    },
    "BNN_hyper_parameters": {
        "hidden_size": [2048, 1014, 512],
        "num_hidden_layers": 3,
        "learning_rate": 0.01,
        "epochs": 10000,
        "dropout": 0.2,
    },
}

NN_MODEL_HSPACE = {
    "NN_hspace": {
        "hidden_size": hp.quniform("hidden_size", 1000, 5000, 1),
        "num_hidden_layers": hp.quniform("num_hidden_layers", 1, 10, 1),
        "learning_rate": hp.loguniform("learning_rate", -7, -2),  # 0.0001 to 0.01
        "epochs": hp.quniform("epochs", 100, 1000, 1),
    }
}
