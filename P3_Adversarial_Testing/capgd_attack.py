import torch
import numpy as np


def _build_mask(feature_names, mutable_features, device):
    """
    Binary mask over features: 1 for mutable, 0 for non-mutable.
    """
    mask = torch.zeros(len(feature_names), device=device)
    for f in mutable_features:
        idx = feature_names.index(f)
        mask[idx] = 1.0
    return mask


def capgd_attack(
    model,
    x,
    x_orig,
    epsilon,
    mask,
    validity_filter,
    steps=20,
    step_frac=0.1,
    device="cpu"
):
    """
    Constrained Adaptive PGD for autoencoder reconstruction loss.
    Operates on a single sample.
    """
    model.eval()

    # Initialize
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad_(True)

    # Step size
    alpha = step_frac * epsilon

    best_x = x_adv.detach().clone()
    best_loss = None

    for _ in range(steps):
        # Forward
        recon = model(x_adv.unsqueeze(0))
        loss = torch.mean((x_adv.unsqueeze(0) - recon) ** 2)

        # Backward
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad

            # Mask gradient (mutable features only)
            grad = grad * mask

            # PGD update
            x_candidate = x_adv + alpha * grad.sign()

            # Project to epsilon-ball around original
            x_candidate = torch.max(
                torch.min(x_candidate, x + epsilon),
                x - epsilon
            )

            # Convert to numpy
            x_candidate_np = x_candidate.cpu().numpy()

            if validity_filter is not None:
                # ----------------------------
                # Constrained mode
                # ----------------------------
                x_proj_np = validity_filter.project(x_candidate_np, x_orig)

                if validity_filter.is_valid(x_proj_np):
                    x_adv = torch.tensor(
                        x_proj_np,
                        dtype=torch.float32,
                        device=device,
                        requires_grad=True
                    )

                    recon_new = model(x_adv.unsqueeze(0))
                    new_loss = torch.mean((x_adv.unsqueeze(0) - recon_new) ** 2)

                    if best_loss is None or new_loss < best_loss:
                        best_loss = new_loss.detach()
                        best_x = x_adv.detach().clone()
                else:
                    # Reject update
                    x_adv = x_adv.detach().clone().requires_grad_(True)

            else:
                # ----------------------------
                # Unconstrained mode (sanity check)
                # ----------------------------
                x_adv = x_candidate.detach().clone().requires_grad_(True)

                recon_new = model(x_adv.unsqueeze(0))
                new_loss = torch.mean((x_adv.unsqueeze(0) - recon_new) ** 2)

                if best_loss is None or new_loss < best_loss:
                    best_loss = new_loss.detach()
                    best_x = x_adv.detach().clone()

        # Reset gradient
        if x_adv.grad is not None:
            x_adv.grad.zero_()

    return best_x.detach()
