import numpy as np
import torch
import torch.nn as nn


# EarlyStopping helper
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if (
            self.best_score is None
            or val_loss < self.best_score - self.min_delta
        ):
            self.best_score = val_loss
            self.best_weights = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")


def make_dataset(input_vals, output_vals, input_dims, output_dims, samples):
    dset_in = []
    dset_out = []
    for i in range(output_dims):
        temp = [input_vals]
        for j in range(i):
            temp.append(np.zeros((samples, 1)))

        temp.append(np.ones((samples, 1)))

        for j in range(i + 1, output_dims):
            temp.append(np.zeros((samples, 1)))

        for j in range(0, i):
            temp.append(output_vals[:, j : j + 1])

        for j in range(i, output_dims - 1):
            temp.append(np.zeros((samples, 1)))

        dset_in.append(np.concatenate(temp, axis=1))
        dset_out.append(output_vals[:, i : i + 1])

    dset_in = np.concatenate(dset_in, axis=0)
    dset_out = np.concatenate(dset_out, axis=0)
    return (dset_in, dset_out)


class QuantileNet(nn.Module):
    def __init__(
        self,
        grad_loss_scale=100,
        tanh_loss_scale=100,
        network_type="normalizing",
        clip=1e-7,
    ):
        super(QuantileNet, self).__init__()
        self.grad_loss_scale = grad_loss_scale
        self.tanh_loss_scale = tanh_loss_scale
        self.clip = clip
        self.net_layers = nn.ModuleList()

        if network_type == "normalizing":
            self.loss_fn = self.normalizing_loss
            self.inner_call = self.normalizing_call
        else:
            self.loss_fn = self.no_normalizing_loss
            self.inner_call = self.no_normalizing_call

    def add(self, layer):
        self.net_layers.append(layer)

    def forward(self, inputs, training=True):
        grad_loss, output_val = self.inner_call(inputs, training=training)
        self._grad_loss = grad_loss
        return output_val

    def normalizing_call(self, inputs):
        count = inputs.shape[0]
        grad_loss = torch.tensor(
            0.0, dtype=torch.float32, device=inputs.device
        )

        # Sample quantiles for median +/- 1 std
        quantiles_low = (
            -2.04 * torch.ones(1, count, device=inputs.device)
        ).pow(1)
        quantiles_mid = torch.zeros(1, count, device=inputs.device)
        quantiles_high = (
            2.04 * torch.ones(1, count, device=inputs.device)
        ).pow(1)

        inputs_low = torch.cat([inputs, quantiles_low], dim=0).T
        inputs_mid = torch.cat([inputs, quantiles_mid], dim=0).T
        inputs_high = torch.cat([inputs, quantiles_high], dim=0).T

        for layer in self.net_layers:
            inputs_low = layer(inputs_low)
            inputs_mid = layer(inputs_mid)
            inputs_high = layer(inputs_high)

        out_low = inputs_low[:, 1:2]
        out_mid = inputs_mid[:, 1:2]
        out_high = inputs_high[:, 1:2]

        shift = out_mid
        scale = (out_high - out_low) / 2

        # Randomly sample quantiles
        quantiles = torch.rand(1, count, device=inputs.device) * 6 - 3
        inputs_q = torch.cat([inputs, quantiles], dim=0).T
        inputs_q.requires_grad_(True)

        val = inputs_q
        for layer in self.net_layers:
            val = layer(val)

        out_norm = val[:, 0:1]
        out_base = val[:, 1:2]

        grads_norm = torch.autograd.grad(
            out_norm.sum(), inputs_q, create_graph=True
        )[0][:, -1]
        loss = (
            self.grad_loss_scale
            * (
                torch.where(
                    grads_norm < 0,
                    grads_norm,
                    torch.tensor(0.0, device=grads_norm.device),
                )
            )
            ** 2
        )
        grad_loss += loss.mean()

        grads_base = torch.autograd.grad(
            out_base.sum(), inputs_q, create_graph=True
        )[0][:, -1]
        loss = (
            self.grad_loss_scale
            * (
                torch.where(
                    grads_base < 0,
                    grads_base,
                    torch.tensor(0.0, device=grads_base.device),
                )
            )
            ** 2
        )
        grad_loss += loss.mean()

        # Transform normalized output
        out_norm = (out_norm + 3) / 6
        abs_out_norm = torch.abs(out_norm)
        loss = self.tanh_loss_scale * torch.where(
            abs_out_norm > 1,
            (abs_out_norm - 1) ** 2,
            torch.tensor(0.0, device=abs_out_norm.device),
        )
        grad_loss += loss.mean()

        output = torch.cat(
            [out_norm, out_base, scale, shift, quantiles.T], dim=1
        )
        return grad_loss, output

    def no_normalizing_call(self, inputs, training=True):
        count = inputs.shape[0]  # batch size
        quantiles = (
            torch.rand(count, 1, device=inputs.device) * 6 - 3
        )  # [B, 1]
        inputs_q = torch.cat([inputs, quantiles], dim=1)  # [B, D+1]
        inputs_q.requires_grad_(True)
        grad_loss = torch.tensor(
            0.0, dtype=torch.float32, device=inputs.device
        )

        val = inputs_q
        for layer in self.net_layers:
            val = layer(val)

        if training:
            out_base = val
            grads = torch.autograd.grad(
                out_base.sum(), inputs_q, create_graph=True
            )[0][:, -1]
            loss = (
                self.grad_loss_scale
                * (
                    torch.where(
                        grads < 0,
                        grads,
                        torch.tensor(0.0, device=grads.device),
                    )
                )
                ** 2
            )
            grad_loss += loss.mean()
        else:
            grad_loss = torch.tensor(0.0, device=inputs.device)

        output = torch.cat([val, quantiles], dim=1)
        return grad_loss, output

    def normalizing_loss(self, y_actual, y_pred):
        quants = (y_pred[:, -1].unsqueeze(1) + 3) / 6
        scale = y_pred[:, 2].unsqueeze(1)
        shift = y_pred[:, 3].unsqueeze(1)

        y_pred_1 = y_pred[:, 0].unsqueeze(1)
        rescaled = (y_actual - shift) / scale
        val = torch.tanh(rescaled) - y_pred_1
        loss_val = torch.where(val < 0, torch.abs(-1 + quants), quants)
        loss_val = loss_val * torch.abs(val)
        true_loss = loss_val.mean()

        y_pred_2 = y_pred[:, 1].unsqueeze(1)
        val = y_actual - y_pred_2
        loss_val = torch.where(val < 0, torch.abs(-1 + quants), quants)
        loss_val = loss_val * torch.abs(val)
        true_loss += loss_val.mean()

        return true_loss

    def no_normalizing_loss(self, y_actual, y_pred):

        quants = (y_pred[:, -1].unsqueeze(1) + 3) / 6
        y_pred_base = y_pred[:, 0].unsqueeze(1)
        val = y_actual - y_pred_base
        loss_val = torch.where(val < 0, torch.abs(-1 + quants), quants)
        loss_val = loss_val * torch.abs(val)
        return loss_val.mean()


def sample_net(
    quantile_object,
    quantile_samples,
    inputs,
    input_count,
    input_dims,
    output_dims,
    network_type="normalizing",
    clip=1e-7,
):
    clip = 1 - clip
    device = inputs.device
    inputs = inputs.T  # (input_count, input_dims)

    def predict_normalizing(val):
        val_low = val[:, :-1].clone()
        val_mid = val[:, :-1].clone()
        val_high = val[:, :-1].clone()

        quantiles_low = -2.04 * torch.ones_like(val[:, -2:-1])
        quantiles_mid = torch.zeros_like(val[:, -2:-1])
        quantiles_high = 2.04 * torch.ones_like(val[:, -2:-1])

        val_low = torch.cat([val_low, quantiles_low], dim=1)
        val_mid = torch.cat([val_mid, quantiles_mid], dim=1)
        val_high = torch.cat([val_high, quantiles_high], dim=1)

        for layer in quantile_object.net_layers:
            val = layer(val)
            val_low = layer(val_low)
            val_mid = layer(val_mid)
            val_high = layer(val_high)

        other_val = val[:, 1:2]
        val = val[:, 0:1]
        val_low = val_low[:, 1:2]
        val_mid = val_mid[:, 1:2]
        val_high = val_high[:, 1:2]

        val = torch.clamp((val + 3) / 6, -clip, clip)
        val = torch.atanh(val) * (val_high - val_low) / 2 + val_mid
        return other_val

    def predict_no_normalizing(val):
        for layer in quantile_object.net_layers:
            val = layer(val)
        return val[:, 0:1]

    predict_inner = (
        predict_normalizing
        if network_type == "normalizing"
        else predict_no_normalizing
    )

    # === Prepare input ===
    sampling_locs = torch.zeros((output_dims, 1), device=device)
    sampling_locs[0] = 1.0
    sampling_locs = sampling_locs.repeat(1, quantile_samples)
    sampling_locs = sampling_locs.repeat(1, input_count)

    random_quant = (
        torch.rand((1, quantile_samples * input_count), device=device) * 6 - 3
    )
    extended_input = inputs.repeat_interleave(quantile_samples, dim=0).T
    zero_inputs = torch.zeros(
        (output_dims - 1, quantile_samples * input_count), device=device
    )

    final_inputs = torch.cat(
        [extended_input, sampling_locs, zero_inputs, random_quant], dim=0
    ).T

    def predict_loop(i, val, output):
        location = i % output_dims
        old_location = (i - 1) % output_dims

        # Build new input vector
        new_input = [val[:, :input_dims]]

        # One-hot encoded output index
        for j in range(output_dims):
            one_hot = (
                torch.ones_like(val[:, 0:1])
                if j == location
                else torch.zeros_like(val[:, 0:1])
            )
            new_input.append(one_hot)

        # Previous outputs
        start = input_dims + output_dims
        mid = start + old_location
        start + output_dims - 1
        new_input.append(val[:, start:mid])  # values before current
        new_input.append(output)  # current prediction
        new_input.append(val[:, mid + 1 : -1])  # values after current

        # New quantiles
        new_quant = torch.rand_like(output) * 6 - 3
        new_input.append(new_quant)

        val = torch.cat(new_input, dim=1)
        output = predict_inner(val)
        return i + 1, val, output

    def predict_main(val):
        output = predict_inner(val)
        i = 1
        while i < output_dims:
            i, val, output = predict_loop(i, val, output)
        # Stitch final vector together
        location = (i - 1) % output_dims
        start = input_dims + output_dims
        mid = start + location
        start + output_dims - 1
        final_output = torch.cat(
            [val[:, start:mid], output, val[:, mid + 1 : -1]], dim=1
        )
        return final_output

    output = predict_main(final_inputs)
    output = output.view(input_count, quantile_samples, output_dims)
    return output
