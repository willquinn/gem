from quantile_network_pytorch import *
from pytorch_train_iqn import load_config, load_data, build_model, setup_logging
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tables
import json
import yaml
import os


def calc_quant(data, vals):
    n = data.shape[1]
    left = np.count_nonzero(vals < data, axis=1)
    right = np.count_nonzero(vals <= data, axis=1)
    pct = (right + left + np.where(right>left, 1+0*right, 0*right)) /(2*n)
    return pct

def load_norm_info(norm_info_path):
    with open(norm_info_path, 'r') as f:
        norm_dict = json.load(f)
    
    norm_info_in = [(entry["mean"], entry["std"]) for entry in norm_dict["input"]]
    norm_info_out = [(entry["mean"], entry["std"]) for entry in norm_dict["output"]]
    return norm_info_in, norm_info_out

# Apply normalization using loaded norm info
def apply_normalization(data, norm_info):
    data_norm = data.copy()
    for i, (mu, sigma) in enumerate(norm_info):
        if sigma == 0:
            data_norm[:, i] = data[:, i] - mu  # avoid division by zero
        else:
            data_norm[:, i] = (data[:, i] - mu) / sigma
    return data_norm

def get_peak_data(samples, input_vals):
    peak_data = {
        "fep": {"Evals": [], "Avals": [], "count": 0, "offset": 0},
        "dep": {"Evals": [], "Avals": [], "count": 0, "offset": 1022},
        "sep": {"Evals": [], "Avals": [], "count": 0, "offset": 511},
    }

    for i in range(len(input_vals)):
        E = input_vals[i]

        for peak, info in peak_data.items():
            E_shifted = E - 2615 + info["offset"]
            if abs(E_shifted) < 5:
                info["count"] += 1
                vals = samples[i, 0::2]  # every second column starting from 0
                info["Evals"].extend(vals.tolist())  # Or use .append(vals) for 2D list
                vals = samples[i, 1::2]  # every second column starting from 0
                info["Avals"].extend(vals.tolist())  # Or use .append(vals) for 2D list

    # Access results
    fep_Evals = peak_data["fep"]["Evals"]
    fep_Avals = peak_data["fep"]["Avals"]
    nfep = peak_data["fep"]["count"]

    dep_Evals = peak_data["dep"]["Evals"]
    dep_Avals = peak_data["dep"]["Avals"]
    ndep = peak_data["dep"]["count"]

    sep_Evals = peak_data["sep"]["Evals"]
    sep_Avals = peak_data["sep"]["Avals"]
    nsep = peak_data["sep"]["count"]

    plt.figure()

    freq, bin_edges = np.histogram(fep_Avals, bins=100, range=(0.4, 1))
    _ = plt.stairs(freq/nfep, bin_edges, label=f'FEP {nfep} events')

    freq, bin_edges = np.histogram(dep_Avals, bins=100, range=(0.4, 1))
    _ = plt.stairs(freq/ndep, bin_edges, label=f'DEP {ndep} events')

    freq, bin_edges = np.histogram(sep_Avals, bins=100, range=(0.4, 1))
    _ = plt.stairs(freq/nsep, bin_edges, label=f'SEP {nsep} events')

    plt.legend()

    # plt.yscale('log')
    plt.xlabel('A/E predicted')
    # plt.ylabel('pdf')

    plt.yscale('log')

    return peak_data

def store_sample_data(model, logger, config, test_in_norm, norm_info_out, test_in):
    output_dims = test_in_norm.shape[1]  # Number of output dimensions
    num_samples = config["sampling"]["num_samples"]
    batch_size = config["sampling"]["batch_size"]

    print(output_dims, num_samples, batch_size)

    # Define HDF5 structure
    filename = config["output"]["sampling_path"]
    h5file = tables.open_file(filename, mode='w')
    atom = tables.Float32Atom()

    # Create extendable array: shape = (0, sample_num * output_dims)
    data_storage = h5file.create_earray(
        h5file.root, 'samples', atom,
        shape=(0, num_samples * output_dims)
    )
    input_storage = h5file.create_earray(
        h5file.root, 'input_reference', atom,
        shape=(0,)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_sample_num = test_in_norm.shape[0]
    for start_idx in range(0, total_sample_num, batch_size):
        end_idx = min(start_idx + batch_size, total_sample_num)
        
        current_test_in = torch.tensor(
            test_in_norm[start_idx:end_idx, :],
            dtype=torch.float32,
            device=device
        )  # shape: (B, input_dims)

        with torch.no_grad():
            out = sample_net(
                model,
                num_samples,
                current_test_in.T,
                current_test_in.shape[0],  # batch_size
                current_test_in.shape[1],  # input_dims
                output_dims,
                network_type="no_normalizing"
            )  # shape: (B, sample_num, output_dims)

        # Convert and denormalize
        out_np = out.detach().cpu().numpy()  # (B, sample_num, output_dims)
        for i in range(output_dims):
            mean_i, std_i = norm_info_out[i]
            out_np[:, :, i] = out_np[:, :, i] * std_i + mean_i

        # Flatten output to (B, sample_num * output_dims)
        reshaped_out = out_np.reshape(out_np.shape[0], -1)

        # Append to HDF5
        data_storage.append(reshaped_out)

        # Optionally store one input column (e.g., index 2)
        current_input = test_in[start_idx:end_idx, 2]
        input_storage.append(current_input)

    # Finalize
    h5file.close()
    logger.info(f"Sampling complete. Data saved to {config['output']['sampling_path']}")

def calc_quant_info(config, logger, samples, test_out):
    output_dims = test_out.shape[1]
    num_samples = config["sampling"]["num_samples"]
    total_samples = samples.shape[0]
    batch_size = config["sampling"]["batch_size"]

    # Pre-allocate full arrays
    median_predictions = np.empty((total_samples, output_dims), dtype=np.float32)
    mean_predictions = np.empty((total_samples, output_dims), dtype=np.float32)
    quantile_errors = np.empty((total_samples, output_dims), dtype=np.float32)

    num_batches = (total_samples + batch_size - 1) // batch_size

    for x in range(num_batches):
        start = x * batch_size
        end = min((x + 1) * batch_size, total_samples)
        actual_batch_size = end - start

        # Slice and reshape
        batch_data = samples[start:end]  # (B, S * D)
        reshaped_data = batch_data.reshape(actual_batch_size, num_samples, output_dims)  # (B, S, D)
        true_data = test_out[start:end, :]  # (B, D)


        for y in range(output_dims):
            sample_y = reshaped_data[:, :, y]     # (B, S)
            true_y = true_data[:, y:y+1]          # (B, 1)

            # Calculate stats
            quant_err = calc_quant(sample_y, true_y)           # (B,)
            median_vals = np.median(sample_y, axis=1)          # (B,)
            mean_vals = np.mean(sample_y, axis=1)              # (B,)

            quantile_errors[start:end, y] = quant_err
            median_predictions[start:end, y] = median_vals
            mean_predictions[start:end, y] = mean_vals

    logger.info("Quantile information calculated.")

    return {
        "median_prediction": median_predictions,
        "mean_prediction": mean_predictions,
        "gen_quantile": quantile_errors,
    }


def summarise_quants(logger, config, test_in, test_out, samples, quant_info_data):

    batch_size = config["sampling"]["batch_size"]
    total_samples = len(samples)
    output_dims = test_out.shape[1]  # Number of output dimensions

    data = {
        'all_counts': [],       # One per output_dim: [sampled_preds (N_bins x N_samples), true_counts]
        'all_edges': [],        # Bin centers for plotting
        'quant_counts': [],     # One per output_dim
        'quant_edges': [],      # Bin edges for quantile hist
        'total_counts': 0,
        'bins': 50,
        'ranges': [[0, 3000], [0, 1]]  #
    }

    num_batches = total_samples // batch_size + int(total_samples % batch_size != 0)

    for x in range(num_batches):
        start = x * batch_size
        end = min((x + 1) * batch_size, total_samples)
        actual_batch_size = end - start

        # Slice data
        temp_ydata = test_out[start:end, :]
        pred_data = samples[start:end, :]
        temp_xdata = test_in[start:end, :]
        quants = quant_info_data['gen_quantile'][start:end, :]

        weights = np.ones(temp_xdata.shape[0])
        data['total_counts'] += np.sum(weights)

        # Reshape pred_data to (B, S, D)
        pred_data = pred_data.reshape(actual_batch_size, -1, output_dims)

        # Compute predicted sample histograms
        pred_counts = []
        for d in range(output_dims):
            dim_preds = pred_data[:, :, d]  # (B, S)
            dim_preds = dim_preds.T         # (S, B) for histogram loop

            counts_per_sample = []
            for s in range(dim_preds.shape[0]):
                counts, _ = np.histogram(dim_preds[s], range=data['ranges'][d], bins=data['bins'], weights=weights)
                counts_per_sample.append(counts)
            counts_per_sample = np.array(counts_per_sample)  # (S, bins)
            pred_counts.append(counts_per_sample)

        # True value histogram
        for d in range(output_dims):
            true_counts, edges = np.histogram(temp_ydata[:, d], range=data['ranges'][d], bins=data['bins'], weights=weights)
            bin_centers = (edges[1:] + edges[:-1]) / 2

            # Quantile histogram (0–1 always)
            q_counts, q_edges = np.histogram(quants[:, d], range=(0, 1), bins=data['bins'], weights=weights)

            if x == 0:
                data['all_counts'].append([pred_counts[d], true_counts])
                data['all_edges'].append(bin_centers)
                data['quant_counts'].append(q_counts)
                data['quant_edges'].append(q_edges[1:])  # right edge for each bin
            else:
                data['all_counts'][d][0] += pred_counts[d]
                data['all_counts'][d][1] += true_counts
                data['quant_counts'][d] += q_counts

    data['pred50'] = []
    data['pred16'] = []
    data['pred84'] = []
    for x in range(output_dims):
        data['pred50'].append(np.quantile(data['all_counts'][x][0], 0.5, axis=0))
        data['pred16'].append(np.quantile(data['all_counts'][x][0], 0.16, axis=0))
        data['pred84'].append(np.quantile(data['all_counts'][x][0], 0.84, axis=0))

    logger.info(f"Total counts: {data['total_counts']}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Finished summarising.")

    return data

def plot_errors(logger, test_out, quant_info_data, output_dir="output/plots"):
    os.makedirs(output_dir, exist_ok=True)

    output_dims = test_out.shape[1]
    assert (
        quant_info_data["median_prediction"].shape[1] == output_dims
    ), "Mismatch in output dimensions"

    median_preds = quant_info_data["median_prediction"]
    mean_preds = quant_info_data["mean_prediction"]
    quant_errors_all = quant_info_data["gen_quantile"]

    for i in range(output_dims):
        true_vals = test_out[:, i]
        median_vals = median_preds[:, i]
        mean_vals = mean_preds[:, i]
        quant_errors = quant_errors_all[:, i]

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Output Dimension {i}: Predictions vs Truth', fontsize=16)

        # Median vs Truth
        axs[0].scatter(true_vals, median_vals, alpha=0.3, label='Median Prediction', s=10)
        axs[0].plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--', label='Ideal')
        axs[0].set_title('Median Prediction vs True')
        axs[0].set_xlabel('True Value')
        axs[0].set_ylabel('Median Prediction')
        axs[0].legend()

        # Mean vs Truth
        axs[1].scatter(true_vals, mean_vals, alpha=0.3, label='Mean Prediction', s=10)
        axs[1].plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--', label='Ideal')
        axs[1].set_title('Mean Prediction vs True')
        axs[1].set_xlabel('True Value')
        axs[1].set_ylabel('Mean Prediction')
        axs[1].legend()

        # Generalized Quantile Error
        axs[2].hist(quant_errors, bins=50, alpha=0.7, color='g')
        axs[2].set_title('Generalized Quantile Error')
        axs[2].set_xlabel('Quantile Error')
        axs[2].set_ylabel('Count')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_dir, f'output_dim_{i}_errors.png')
        plt.savefig(plot_path)
        plt.close(fig)

        logger.info(f"Saved error plots for output dimension {i} to: {plot_path}")

    # Relative error plots
    ranges = [(-0.25, 0.25), (-1, 1)]
    titles = ["Energy", "AoEs"]
    for i in range(output_dims):
        pred_median = median_preds[:, i]
        true_vals = test_out[:, i]

        # Handle NaNs (should be rare)
        valid = ~np.isnan(pred_median) & (true_vals != 0)
        rel_error = np.zeros_like(pred_median)
        rel_error[valid] = (pred_median[valid] - true_vals[valid]) / true_vals[valid]

        fig = plt.figure(figsize=(7, 4))
        plt.hist(rel_error, range=ranges[i], bins=100, histtype='step', label=f'Rel. Error (dim {i})')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel("Relative Error")
        plt.ylabel("Count")
        plt.title(f"Relative Error for {titles[i]}")
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'output_dim_{i}_relative_error.png')
        plt.savefig(plot_path)
        plt.close(fig)

        logger.info(f"Relative error plot for output dimension {i} saved: output/plots/output_dim_{i}_relative_error.png")
        
def plot_quality(logger, output_dims, quant_summary_data):
    titles = ["E", "A/E"]
    bins = quant_summary_data['bins']

    for x in range(output_dims):
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 3))

        # Get quantile histogram info
        counts = quant_summary_data['quant_counts'][x]  # shape: (bins,)
        edges = quant_summary_data['quant_edges'][x]   # shape: (bins,)
        centers = (edges[1:] + edges[:-1]) / 2 if len(edges) == bins + 1 else edges

        # Normalize
        normalized = counts / np.sum(counts)

        # Plot predicted quantile distribution
        ax1.plot(centers, normalized, "o", color="#d7301f", label="predicted")
        ax1.axhline(1.0 / bins, color="k", linestyle="--", label="uniform target")

        ax1.set_ylabel("p(quantile)")
        ax1.set_xlabel("quantile")
        ax1.set_title(titles[x])
        ax1.set_ylim(0,0.1)
        ax1.legend()
        plt.tight_layout()
        plt.savefig(f'output/plots/output_dim_{x}_quantile_quality.png')
        logger.info(f"Quantile quality plot for output dimension {x} saved: output/plots/output_dim_{x}_quantile_quality.png")

def plot_comparison(logger, output_dims, qs_data):
    labels = ["E", "A/E"]
    for x in range(output_dims): 
        edges = qs_data['all_edges'][x] 
        all_counts = qs_data['all_counts'][x]   
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
        
        ax1.step(edges, all_counts[1], where="mid", color="k", linewidth=0.5)#, linestyle="-.")
        ax1.step(edges, qs_data['pred50'][x], where="mid", color="#d7301f", linewidth=0.5)
        ax1.scatter(edges, qs_data['pred50'][x], label="IQN median", color="#d7301f", marker="x", s=5, linewidth=0.5)
        ax1.scatter(edges, all_counts[1], label="PSS",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
        
        ax1.set_xlim(qs_data['ranges'][x])
        # ax1.set_ylim(0, max(all_counts[x][0])*1.1)
        ax1.set_ylabel("counts")
        ax1.set_xticklabels([])
        ax1.legend()

        ax1.set_yscale('log')
        
        ax2.scatter(edges, all_counts[1]/all_counts[1], color="k", marker="o",facecolors="none", s=5, linewidth=0.5)
        ax2.errorbar(
            edges, qs_data['pred50'][x]/all_counts[1],
            xerr=0, yerr=[qs_data['pred50'][x]/all_counts[1]- qs_data['pred16'][x]/all_counts[1], qs_data['pred84'][x]/all_counts[1]-qs_data['pred50'][x]/all_counts[1]],
            color="#d7301f", ls="",
            capsize=2,capthick=0.5, marker="x", linewidth=0.5, markersize=np.sqrt(5)
        )

        ax2.set_xlabel(labels[x])
        #ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{gen}}$")
        # ax2.set_ylim((0.9,1.1))
        ax2.set_xlim(qs_data['ranges'][x])
        plt.savefig(f'output/plots/output_dim_{x}_comparison.png')
        logger.info(f"Comparison plot for output dimension {x} saved: output/plots/output_dim_{x}_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Train IQN with PyTorch")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--log-level', type=str, default="INFO", help='Override log level (e.g., DEBUG, INFO, WARNING)')
    parser.add_argument('--plotting', type=bool, default=False, help='Enable plotting of training data')
    args = parser.parse_args()

    config = load_config(args.config)

    logger = setup_logging(config)
    logger.info("Starting sampling run")
    logger.info("Parsed arguments: %s", args)
    logger.info("Full configuration:\n%s", json.dumps(config, indent=2))

    input_dims_amend = len(config['data']['x']) + 2 + len(config['data']['y']) - 1 # 2 from one hot encoding, n-1 for tragets and 1 for quantiles 
    model = build_model(config, input_dims_amend)
    model.load_state_dict(torch.load(config['output']['model_path']))
    model.eval()

    test_idx  = np.load(config["output"]["test_idx"])
    train_idx  = np.load(config["output"]["train_idx"])
    norm_info_in, norm_info_out = load_norm_info(config["output"]["norm_path"])

    xdata, ydata = load_data(config)
    train_in = xdata[train_idx]
    train_out = ydata[train_idx]
    test_in = xdata[test_idx]
    test_out = ydata[test_idx]

    test_in_norm = apply_normalization(test_in, norm_info_in)
    test_out_norm = apply_normalization(test_out, norm_info_out)
    output_dims = test_out_norm.shape[1]

    if os.path.exists(config["output"]["sampling_path"]):
        logger.info("Sample data already exists, loading from file.")
    else:
        logger.info("Sample data not found, generating new samples.")
        # Ensure the directory exists
        store_sample_data(model, logger, config, test_in_norm, norm_info_out, test_in) # closes file
        
    with tables.open_file(config["output"]["sampling_path"], mode='r') as f:
        samples = f.root.samples[:]
        input_vals = f.root.input_reference[:]

    quant_info_data = calc_quant_info(config, logger, samples, test_out)
    plot_errors(logger, test_out, quant_info_data)

    quant_summary = summarise_quants(logger, config, test_in_norm, test_out_norm, samples, quant_info_data)
    plot_quality(logger, output_dims, quant_summary)

    plot_comparison(logger, output_dims, quant_summary)


if __name__ == "__main__":
    main()

