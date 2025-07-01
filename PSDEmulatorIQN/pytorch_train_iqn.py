import argparse
import glob
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from quantile_network_pytorch import EarlyStopping, QuantileNet, make_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def setup_logging(config, path_overide=None):
    os.makedirs("output/logs", exist_ok=True)
    logger = logging.getLogger()

    log_level = config["output"].get("log_level", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    if path_overide is not None:
        fh = logging.FileHandler(path_overide)
    else:
        fh = logging.FileHandler(config["output"]["log_path"])
    fh.setLevel(numeric_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_data(config, logger=None):
    dfs = []
    if logger is not None:
        logger.debug(
            f"Looking for files matching: {config['data']['location']}"
        )

    file_list = sorted(glob.glob(config["data"]["location"]))

    if not file_list and logger is not None:
        logger.error(
            f"No files found for pattern: {config['data']['location']}"
        )
        raise FileNotFoundError(
            f'No files found for pattern: {config["data"]["location"]}'
        )

    for filepath in file_list:
        if logger is not None:
            logger.info(f"Loading {filepath}")
        df = pd.read_json(filepath).set_index("ievt_n")
        if df.isnull().values.any() and logger is not None:
            logger.warning(f"NaNs found in {filepath}, dropping them")
        df = df.dropna()
        dfs.append(df)

    df_data = pd.concat(dfs, axis=0).dropna()

    xdata = df_data[config["data"]["x"]].to_numpy()
    ydata = df_data[config["data"]["y"]].to_numpy()

    return xdata, ydata


def load_and_prepare_data(config, logger):

    xdata, ydata = load_data(config, logger)

    logger.info(
        f"Data loaded with shapes - X: {xdata.shape}, Y: {ydata.shape}"
    )
    logger.debug(f"xdata columns: {config['data']['x']}")
    logger.debug(f"ydata columns: {config['data']['y']}")
    logger.info(f"Data loaded: {len(xdata)} total events")

    # Determine full dataset indices
    all_indices = np.arange(len(xdata))
    logger.info("Splitting train/test using seed %d", config["random_seed"])
    # Perform split using indices
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=config["training"]["test_size"],
        random_state=config["random_seed"],
    )
    logger.debug(
        f"Train indices: {train_idx[:5]}... Test indices: {test_idx[:5]}..."
    )

    # Save indices for reproducibility
    np.save(config["output"]["train_idx"], train_idx)
    np.save(config["output"]["test_idx"], test_idx)
    logger.info(
        "Train/test indexes saved:"
        + f" {config['output']['train_idx']}, {config['output']['test_idx']}"
    )

    # Split data using the saved indices
    train_in, test_in = xdata[train_idx], xdata[test_idx]
    train_out, test_out = ydata[train_idx], ydata[test_idx]

    logger.info(
        "Train/test split done. Train X shape:"
        + f" {train_in.shape}, Test X shape: {test_in.shape}"
    )

    return train_in, test_in, train_out, test_out


def normalise_data(train_in, test_in, train_out, test_out, logger, config):
    # Normalise input/output and store means/stds
    input_dims = train_in.shape[1]
    output_dims = train_out.shape[1]

    norm_info_in = []
    norm_info_out = []

    train_in_norm = train_in.copy()
    test_in_norm = test_in.copy()
    train_out_norm = train_out.copy()
    test_out_norm = test_out.copy()

    for i in range(input_dims):
        mu = np.mean(train_in[:, i])
        sigma = np.std(train_in[:, i])
        if sigma == 0:
            logger.warning(
                f"Input feature {i}"
                + " has zero std — may indicate constant feature."
            )
        logger.debug(f"Input feature {i}: μ={mu:.3f}, σ={sigma:.3f}")
        norm_info_in.append((mu, sigma))
        train_in_norm[:, i] = (train_in[:, i] - mu) / sigma
        test_in_norm[:, i] = (test_in[:, i] - mu) / sigma

    for i in range(output_dims):
        mu = np.mean(train_out[:, i])
        sigma = np.std(train_out[:, i])
        logger.debug(f"Output target {i}: μ={mu:.3f}, σ={sigma:.3f}")

        norm_info_out.append((mu, sigma))
        train_out_norm[:, i] = (train_out[:, i] - mu) / sigma
        test_out_norm[:, i] = (test_out[:, i] - mu) / sigma

    norm_dict = {
        "input": [
            {"mean": float(mu), "std": float(sigma)}
            for mu, sigma in norm_info_in
        ],
        "output": [
            {"mean": float(mu), "std": float(sigma)}
            for mu, sigma in norm_info_out
        ],
    }
    with open(config["output"]["norm_path"], "w") as f:
        json.dump(norm_dict, f, indent=4)
    logger.info(f"Normalization info saved to {config['output']['norm_path']}")

    return (
        train_in_norm,
        test_in_norm,
        train_out_norm,
        test_out_norm,
        norm_info_in,
        norm_info_out,
    )


def prepare_iqn_datasets(train_in_norm, train_out_norm, config, logger):
    input_dims = train_in_norm.shape[1]
    output_dims = train_out_norm.shape[1]
    n_samples = train_in_norm.shape[0]

    x_val, y_val = make_dataset(
        train_in_norm, train_out_norm, input_dims, output_dims, n_samples
    )
    logger.info(
        "make_dataset returned shapes:"
        + f" x_val {x_val.shape}, y_val {y_val.shape}"
    )

    iqn_train_in, iqn_val_in, iqn_train_out, iqn_val_out = train_test_split(
        x_val,
        y_val,
        test_size=config["training"]["val_size"],
        random_state=config["random_seed"],
    )

    # Convert to torch tensors
    iqn_train_in = torch.tensor(iqn_train_in, dtype=torch.float32)
    iqn_val_in = torch.tensor(iqn_val_in, dtype=torch.float32)
    iqn_train_out = torch.tensor(iqn_train_out, dtype=torch.float32)
    iqn_val_out = torch.tensor(iqn_val_out, dtype=torch.float32)

    return iqn_train_in, iqn_val_in, iqn_train_out, iqn_val_out


def build_model(config, input_dims):
    model = QuantileNet(network_type=config["model"]["network_type"])

    # Add layers
    model.add(
        nn.Linear(
            in_features=input_dims + 1,
            out_features=config["structure"]["width"],
        )
    )
    model.add(nn.LeakyReLU())

    for _ in range(config["structure"]["hidden"] - 1):
        model.add(
            nn.Linear(
                in_features=config["structure"]["width"],
                out_features=config["structure"]["width"],
            )
        )
        model.add(nn.LeakyReLU())

    model.add(
        nn.Linear(
            in_features=config["structure"]["width"],
            out_features=config["structure"]["layer_output_dims"],
        )
    )

    return model


def train_loop(model, train_loader, val_loader, config, logger):
    optimizer = optim.Adam(
        model.parameters(), lr=config["training"]["initiallr"], amsgrad=False
    )
    early_stopping = EarlyStopping(patience=config["training"]["patience"])

    train_loss_history = []
    val_loss_history = []
    epoch_list = []

    total_epochs = 0

    for cycle in range(config["training"]["cycles"]):
        lr = config["training"]["initiallr"] * (10**-cycle)
        logger.info(f"Starting training cycle {cycle+1} with LR={lr:.1e}")

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for epoch in range(config["training"]["epochs"]):
            model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_x, training=True)
                loss = model.loss_fn(batch_y, preds)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_x, val_y in val_loader:
                    val_preds = model(val_x, training=False)
                    val_loss = model.loss_fn(val_y, val_preds)
                    val_losses.append(val_loss.item())

            mean_train_loss = np.mean(train_losses)
            mean_val_loss = np.mean(val_losses)

            train_loss_history.append(mean_train_loss)
            val_loss_history.append(mean_val_loss)
            epoch_list.append(total_epochs + 1)

            logger.debug(
                f"Epoch {epoch+1}:"
                + f" mean train loss = {mean_train_loss:.4f},"
                + f" mean val loss = {mean_val_loss:.4f}"
            )

            early_stopping(mean_val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                model.load_state_dict(early_stopping.best_weights)
                break

            total_epochs += 1

        if early_stopping.early_stop:
            logger.info("Early stopping triggered, restoring best weights")
            break

    return model, train_loss_history, val_loss_history, epoch_list


def plot_training_distributions(
    train_in,
    train_out,
    train_in_norm,
    train_out_norm,
    iqn_train_in,
    iqn_train_out,
    logger,
):

    # Raw input/output features
    for label, data in zip(["in", "out"], [train_in, train_out]):
        for i in range(data.shape[1]):
            plt.figure()
            plt.hist(data[:, i], bins=300)
            plt.xlabel(f"{label.upper()} Feature {i}")
            plt.ylabel("Counts")
            path = f"plots/train_{label}_feature_{i}.png"
            plt.savefig(path)
            plt.close()
            logger.infor(f"Saved raw feature plot: {path}")

    # Normalized input/output features
    for label, data in zip(["in", "out"], [train_in_norm, train_out_norm]):
        for i in range(data.shape[1]):
            plt.figure()
            plt.hist(data[:, i], bins=300)
            plt.xlabel(f"Normalized {label.upper()} Feature {i}")
            plt.ylabel("Counts")
            path = f"plots/train_norm_{label}_feature_{i}.png"
            plt.savefig(path)
            plt.close()
            logger.info(f"Saved normalized feature plot: {path}")

    # One-hot encoded feature breakdown (hardcoded indices 5 and 6)
    for one_hot_index in [5, 6]:
        mask = np.where(iqn_train_in[:, one_hot_index] == 1)[0]

        if len(mask) == 0:
            logger.info(
                "No entries found for one-hot index"
                + f" {one_hot_index}, skipping..."
            )
            continue

        # Histogram of input feature 2
        plt.figure()
        plt.hist(iqn_train_in[mask, 2], bins=300)
        plt.xlabel("Input Feature 2")
        plt.ylabel("Counts")
        path = f"plots/iqn_input_feature_2_ohe_{one_hot_index}.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved one-hot input histogram: {path}")

        # Histogram of output
        plt.figure()
        plt.hist(iqn_train_out[mask], bins=300)
        plt.xlabel("Output Feature")
        plt.ylabel("Counts")
        path = f"plots/iqn_output_feature_ohe_{one_hot_index}.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved one-hot output histogram: {path}")

        # Scatter plot input vs output
        plt.figure()
        plt.scatter(iqn_train_in[mask, 2], iqn_train_out[mask], alpha=0.2)
        plt.xlabel("Input Feature 2")
        plt.ylabel("Output Feature")
        path = f"plots/iqn_input_vs_output_ohe_{one_hot_index}.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved input vs output scatter: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train IQN with PyTorch")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Override log level (e.g., DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--plotting",
        type=bool,
        default=False,
        help="Enable plotting of training data",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    logger = setup_logging(config)
    logger.info("Starting training run")
    logger.debug("Parsed arguments: %s", args)
    logger.info(
        "Full configuration:\n%s", yaml.dump(config, default_flow_style=False)
    )

    # Set seeds for reproducibility
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    train_in, test_in, train_out, test_out = load_and_prepare_data(
        config, logger
    )
    (
        train_in_norm,
        test_in_norm,
        train_out_norm,
        test_out_norm,
        norm_info_in,
        norm_info_out,
    ) = normalise_data(train_in, test_in, train_out, test_out, logger, config)
    iqn_train_in, iqn_val_in, iqn_train_out, iqn_val_out = (
        prepare_iqn_datasets(train_in_norm, train_out_norm, config, logger)
    )

    if args.plotting:
        plot_training_distributions(
            train_in,
            train_out,
            train_in_norm,
            train_out_norm,
            iqn_train_in,
            iqn_train_out,
            logger,
        )

    model = build_model(config, iqn_train_in.shape[1])
    logger.info("Model initialized")
    logger.debug("Model structure:\n%s", model)

    train_dataset = TensorDataset(iqn_train_in, iqn_train_out)
    val_dataset = TensorDataset(iqn_val_in, iqn_val_out)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    model, train_loss, val_loss, epoch_list = train_loop(
        model, train_loader, val_loader, config, logger
    )

    # Save model weights
    torch.save(model.state_dict(), config["output"]["model_path"])
    logger.info(f"Model saved to {config['output']['model_path']}")

    # Combine into a DataFrame
    loss_df = pd.DataFrame(
        {"epoch": epoch_list, "train_loss": train_loss, "val_loss": val_loss}
    )

    # Save CSV
    loss_df.to_csv(config["output"]["loss_csv"], index=False)

    # Log summary to log file
    logger.info("Final losses:")
    for epoch, tr_loss, vl_loss in zip(epoch_list, train_loss, val_loss):
        logger.info(
            "Epoch %3d | Train Loss: %.6f | Val Loss: %.6f",
            epoch,
            tr_loss,
            vl_loss,
        )

    logger.info("Loss history saved to: %s", config["output"]["loss_csv"])


if __name__ == "__main__":
    main()
