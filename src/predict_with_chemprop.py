from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold
import tempfile
import os

from modules.data_load import load_reaction_data
from predict_settings import data_info

# ChemProp 2.x imports
try:
    import chemprop
    from chemprop.data import MoleculeDataset, build_dataloader
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    CHEMPROP_AVAILABLE = True
except ImportError as e:
    print(f"ChemProp not available: {e}")
    print("Please install chemprop: conda install chemprop-rdkit -c conda-forge")
    CHEMPROP_AVAILABLE = False


def prepare_chemprop_data(data_df, reagent_columns, target_type):
    """
    Prepare SMILES data for ChemProp training
    For ULD dataset: Use only valid organic molecule SMILES, not reaction SMILES
    """
    from rdkit import Chem

    smiles_data = []
    targets = []

    for _, row in data_df.iterrows():
        # Collect valid organic molecule SMILES (excluding metals and blank cells)
        valid_smiles = []

        for col in reagent_columns:
            if pd.notna(row[col]) and row[col] != "blank_cell":
                # Try to parse with RDKit to check validity
                try:
                    mol = Chem.MolFromSmiles(row[col])
                    if mol is not None:
                        # Clean the SMILES
                        clean_smi = Chem.MolToSmiles(mol)
                        valid_smiles.append(clean_smi)
                except:
                    print(row[col], "is not a valid SMILES")
                    continue

            # Combine valid SMILES with dot notation (mixture representation)

        combined_smiles = ".".join(valid_smiles)  # Limit to first 4 molecules to avoid too complex
        smiles_data.append(combined_smiles)
        targets.append(row[target_type])

    print(f"Valid SMILES examples:")
    for i in range(min(3, len(smiles_data))):
        print(f"  {smiles_data[i]}")

    return smiles_data, targets


def create_chemprop_dataset(smiles_data, targets=None):
    """
    Create ChemProp 2.x MoleculeDataset
    """
    from chemprop.data import MoleculeDatapoint
    import numpy as np

    datapoints = []
    for i, smi in enumerate(smiles_data):
        if targets is not None:
            # Create datapoint with target
            y = np.array([targets[i]], dtype=np.float32)
            datapoint = MoleculeDatapoint.from_smi(smi, y=y)
        else:
            # Create datapoint without target (for prediction)
            datapoint = MoleculeDatapoint.from_smi(smi)
        datapoints.append(datapoint)

    dataset = MoleculeDataset(datapoints)
    return dataset


class ChemPropModel(nn.Module):
    """
    ChemProp 2.x model wrapper using proper components
    """

    def __init__(self, hidden_size=300, depth=3, dropout=0.1):
        super().__init__()

        from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

        # Message passing component
        self.message_passing = BondMessagePassing(d_h=hidden_size, depth=depth, dropout=dropout, activation="relu")

        # Aggregation component
        self.agg = MeanAggregation()

        # Predictor component
        self.predictor = RegressionFFN(input_dim=hidden_size, hidden_dim=hidden_size, n_layers=2, dropout=dropout)

        # Create the full MPNN model
        from chemprop.models import MPNN

        self.mpnn = MPNN(message_passing=self.message_passing, agg=self.agg, predictor=self.predictor)

    def forward(self, batch):
        # Extract the molecular graph from the batch
        return self.mpnn(batch.bmg, batch.V_d, batch.X_d)


def train_chemprop_model_v2(train_dataset, val_dataset, model_dir, epochs=50):
    """
    Train ChemProp 2.x model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_dataloader = build_dataloader(train_dataset, batch_size=32, shuffle=True)

    # Create model
    model = ChemPropModel().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            # ChemProp 2.x batch handling - targets are in batch.Y
            targets = batch.Y.to(device)

            optimizer.zero_grad()
            predictions = model(batch)

            loss = criterion(predictions.squeeze(), targets.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

    return model


def predict_chemprop_model_v2(model_path, test_dataset):
    """
    Make predictions using ChemProp 2.x model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ChemPropModel().to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=device))
    model.eval()

    # Create data loader
    test_dataloader = build_dataloader(test_dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            # No need for batch.to(device) - handle device in model forward
            pred = model(batch)
            predictions.extend(pred.squeeze().cpu().numpy())

    return np.array(predictions)


def train_models_chemprop(X_data, y_data, dataset_type, split_info, target):
    """
    Train ChemProp models with cross-validation using ChemProp 2.x API
    """
    all_train_preds = []
    all_train_targets = []
    all_test_preds = []
    all_test_targets = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_info):
        print(f"Training fold {fold_idx + 1}...")

        # Split data
        train_smiles = [X_data[i] for i in train_idx]
        train_targets = [y_data[i] for i in train_idx]
        test_smiles = [X_data[i] for i in test_idx]
        test_targets = [y_data[i] for i in test_idx]

        # Further split training data for validation
        val_size = int(0.1 * len(train_smiles))
        val_smiles = train_smiles[:val_size]
        val_targets = train_targets[:val_size]
        train_smiles = train_smiles[val_size:]
        train_targets = train_targets[val_size:]

        # Create datasets
        train_dataset = create_chemprop_dataset(train_smiles, train_targets)
        val_dataset = create_chemprop_dataset(val_smiles, val_targets)
        test_dataset = create_chemprop_dataset(test_smiles, test_targets)  # Need targets for consistency

        # Create model directory for this fold
        model_dir = Path(__file__).parent / f"chemprop_models/{dataset_type}_fold_{fold_idx}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        model = train_chemprop_model_v2(train_dataset, val_dataset, str(model_dir), epochs=50)

        # Make predictions
        train_val_dataset = create_chemprop_dataset(train_smiles + val_smiles, train_targets + val_targets)
        train_preds = predict_chemprop_model_v2(str(model_dir), train_val_dataset)
        test_preds = predict_chemprop_model_v2(str(model_dir), test_dataset)

        # Store results
        all_train_preds.extend(train_preds)
        all_train_targets.extend(train_targets + val_targets)
        all_test_preds.extend(test_preds)
        all_test_targets.extend(test_targets)

    # Create results DataFrames
    train_results = pd.DataFrame({"real": all_train_targets, "predict": all_train_preds})

    test_results = pd.DataFrame({"real": all_test_targets, "predict": all_test_preds})

    # Save results
    train_results.to_csv(Path(__file__).parent / f"logs/{dataset_type}/chemprop_{target}_train.csv", index=False)
    test_results.to_csv(Path(__file__).parent / f"logs/{dataset_type}/chemprop_{target}_test.csv", index=False)

    return train_results, test_results


def generate_split_info_chemprop(data_df, split_mode, n_splits=10):
    """
    Generate split information for cross-validation
    """
    if split_mode == "random":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(kf.split(data_df))
    else:
        # Add other split modes as needed
        raise ValueError(f"Split mode {split_mode} not implemented")


def dataset_prediction_chemprop(dataset_type="ULD", target_type="yield", split_mode="random"):
    """
    Main function for ChemProp-based prediction
    """
    if not CHEMPROP_AVAILABLE:
        print("ChemProp is not available. Please install it first.")
        return

    # Create logs directory
    log_dir = Path(__file__).parent / f"logs/{dataset_type}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load reaction data
    data_df = load_reaction_data(dataset_type, target_type)
    reagent_columns = data_info[dataset_type]["reagent_columns"]

    # Prepare SMILES data for ChemProp
    smiles_data, targets = prepare_chemprop_data(data_df, reagent_columns, target_type)

    print(f"Prepared {len(smiles_data)} reactions for ChemProp training")

    # Generate split information
    n_splits = 10
    split_info = generate_split_info_chemprop(pd.DataFrame({"smiles": smiles_data, "target": targets}), split_mode, n_splits)

    # Train models
    print("Training ChemProp models...")
    train_results, test_results = train_models_chemprop(smiles_data, targets, dataset_type, split_info, target_type)

    # Calculate metrics
    r2_train = r2_score(train_results["real"], train_results["predict"])
    r2_test = r2_score(test_results["real"], test_results["predict"])
    mae_train = mean_absolute_error(train_results["real"], train_results["predict"])
    mae_test = mean_absolute_error(test_results["real"], test_results["predict"])
    rmse_train = root_mean_squared_error(train_results["real"], train_results["predict"])
    rmse_test = root_mean_squared_error(test_results["real"], test_results["predict"])

    print(f"for ChemProp model,")
    print(f"    r2_train: {r2_train:.2f}, r2_test: {r2_test:.2f}")
    print(f"    mae_train: {mae_train:.2f}, mae_test: {mae_test:.2f}")
    print(f"    rmse_train: {rmse_train:.2f}, rmse_test: {rmse_test:.2f}")


if __name__ == "__main__":
    dataset_prediction_chemprop(dataset_type="ULD", target_type="yield", split_mode="random")
