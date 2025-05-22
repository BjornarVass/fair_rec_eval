import torch
import numpy as np

from optin.model import EvalStorage
from optin.utils import evaluate_all_recommendations, update_early_stopping


def info_align_encode(model, s, orig_latent, device, mask):
    feature_encodings = []
    combined = torch.zeros((s.shape[0], 0), device=device)
    for j in range(model.c.n_sensitive + 1):
        if j == 0 or not model.sim_info_align:
            encoded_latent = model.align_encode[j](orig_latent.detach())
            feature_encodings.append(encoded_latent)
        # One-hot encodings of each sensitive label: (0=no, 1=yes, 2=unknown)
        if j > 0 and model.sim_info_align:
            encoded_latent = torch.zeros((s.shape[0], 3), dtype=torch.float32, device=device)
            if mask[j - 1] == 1:
                encoded_latent[:, 2] = 1
            else:
                encoded_latent[torch.arange(s.shape[0]), s.to(torch.int64)[:, j - 1]] = 1
        combined = torch.cat((combined, encoded_latent), 1)
    return combined, feature_encodings


# Note: BPR loss is a poor match for VAE recommenders. We kept the implementation for the sake of reproducability
def get_bpr_samples(x, device):
    pos_indices = torch.zeros((x.shape[0], 1), dtype=torch.int64, device=device)
    neg_indices = torch.zeros((x.shape[0], 1), dtype=torch.int64, device=device)
    for i in range(x.shape[0]):
        pos = x[i][x[i] == 1.0]
        pos_ind = np.random.randint(pos.shape[0])
        pos_indices[i] = pos_ind
        neg = x[i][x[i] == 0.0]
        neg_ind = np.random.randint(neg.shape[0])
        neg_indices[i] = neg_ind
    return pos_indices, neg_indices


def update_encoders(
    model, settings, latent, s, feature_encodings, beta, encode_optim, class_optim, neut_encode_optim, adv_optim
):
    model = settings.model
    afrl = settings.afrl

    # Calculate loss for encoders and
    feature_loss = 0
    for j in range(len(model.align_encode)):
        if j != 0 and not model.sim_info_align:
            loss_mask = np.zeros(model.c.n_sensitive)
            loss_mask[j - 1] = 1
            target_loss = model.discriminate_simple(feature_encodings[j], s, loss_weight=loss_mask, adv=False)
            norm_loss = torch.norm(feature_encodings[j], dim=1).mean()
            feature_loss += target_loss + beta * norm_loss
        else:
            adv_loss = model.discriminate_simple(feature_encodings[0], s)
            target_loss = torch.nn.functional.mse_loss(feature_encodings[0], latent.detach())
            ce_loss = target_loss - afrl.lambd * adv_loss

    # Update sensitive encodings and classifiers if in use
    if not model.sim_info_align:
        encode_optim.zero_grad()
        class_optim.zero_grad()
        feature_loss.backward()
        encode_optim.step()
        class_optim.step()

    # Update neutral encoder
    neut_encode_optim.zero_grad()
    ce_loss.backward()
    neut_encode_optim.step()

    # Update adversarials
    for _ in range(settings.n_adv_train):
        adv_optim.zero_grad()
        disc_loss = model.discriminate_simple(feature_encodings[0].detach(), s)
        disc_loss.backward()
        adv_optim.step()
    return target_loss, adv_loss


def update_combination_network(
    x, latent, final_latents, logits, loss, combine_optim, combine_bpr, combine_mse, device
):
    # "loss" is currently the decoder loss, may be overridden by bpr or mse_loss
    combine_optim.zero_grad()
    if combine_bpr:
        pos, neg = get_bpr_samples(x, device)
        pos_logs = logits.gather(1, pos.detach())
        neg_logs = logits.gather(1, neg.detach())
        loss = torch.nn.functional.softplus(-(pos_logs - neg_logs)).mean()
    elif combine_mse:
        loss = torch.nn.functional.mse_loss(final_latents, latent.detach())

    loss.backward()
    combine_optim.step()
    return loss


def debug_afrl(settings, epoch, n_encode_epochs, x_val, s_val, y_val, loss, target_loss, adv_loss):
    model = settings.model
    device = model.c.device
    model.eval()
    with torch.no_grad():
        lat_val, _, _ = model(x_val, s_val, {}, decode=False)
        masks = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        enc_lat = model.align_encode[0](lat_val)
        adv_loss_val = model.discriminate_simple(enc_lat, s_val)
        target_loss_val = torch.nn.functional.mse_loss(enc_lat, lat_val.detach())
        if epoch > n_encode_epochs:
            prnt_msg = f"tot: {loss.item()}"
            for j in range(4):
                eval_storage = EvalStorage()
                combined_val, _ = info_align_encode(model, s_val, lat_val, device, masks[j])
                fin_lat_val = model.combine_net(combined_val)
                probs_val, _ = model.decode(x_val, fin_lat_val)
                eval_storage.process_and_update(probs_val, x_val, y_val, fin_lat_val, s_val, False)
                eval_storage.concat()
                eval_res = evaluate_all_recommendations(settings, eval_storage.u_probs, eval_storage.targets)
                prnt_msg += f", ndcg{j}: {eval_res.ndcg}"
            print(prnt_msg)
    if epoch <= n_encode_epochs:
        print(
            f"tot: {loss.item()}, mse: {target_loss.item()}, mse val: {target_loss_val.item()}, adv: {adv_loss.item()}, adv val: {adv_loss_val.item()}"
        )


# We based our implementation on the official implementation found here:
# https://github.com/zhuxinyu2700/AFRL
# See the paper for details on what we had to change, adapt or do different for our VAE + user-split setting.
def train_info_align(settings):
    model = settings.model
    afrl = settings.afrl

    # Settings used in all exps. Turning off MSE loss will enable the decoder multinomial loss
    # Also turning on bpr will enable BPR loss like in: https://github.com/zhuxinyu2700/AFRL (see comments in paper)
    combine_mse = True
    combine_bpr = False

    # Baseline can overfit all parts except for the combination since embeddings are known
    # for all users
    comb_epochs = afrl.n_encode_epochs + afrl.n_combine_epochs

    # Build optimizers
    neut_encode_optim = torch.optim.Adam(
        model.align_encode[0].parameters(), lr=afrl.lr, weight_decay=afrl.weight_decay
    )
    # Skip encoder and classifier optimizers for each sensiteive attribute if we use simple embeddings
    encode_optim = None
    class_optim = None
    if not model.sim_info_align:
        encode_params = []
        class_params = []
        for i in range(model.c.n_sensitive):
            encode_params += model.align_encode[1 + i].parameters()
            class_params += model.classifiers[i].parameters()
        encode_optim = torch.optim.Adam(encode_params, lr=afrl.lr, weight_decay=afrl.weight_decay)
        class_optim = torch.optim.Adam(class_params, lr=afrl.lr, weight_decay=afrl.weight_decay)
    adv_optim = torch.optim.Adam(model.get_params(adversarial=True), lr=afrl.lr, weight_decay=afrl.weight_decay)
    combine_optim = torch.optim.Adam(model.combine_net.parameters(), lr=afrl.lr, weight_decay=afrl.weight_decay)

    # Extract validation data
    device = model.c.device
    x_val = settings.val_loader.dataset.x
    y_val = settings.val_loader.dataset.y
    s_val = settings.val_loader.dataset.s
    if settings.csr:
        x_val = torch.tensor(x_val.toarray(), dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val.toarray(), dtype=torch.float32)
        s_val = torch.tensor(s_val, dtype=torch.float32, device=device)
    else:
        y_val = torch.tensor(y_val, dtype=torch.float32)

    model.train()
    n_es = settings.early_stopping
    early_stopping = n_es > 0
    min_epochs = 300
    es_counter = 0
    epoch = 0
    best_loss = np.inf
    eps = 1e-4
    best_model = None
    upper_epoch = comb_epochs if not early_stopping else 3000
    train_comb = False
    while epoch < upper_epoch:
        epoch_loss = []
        if not early_stopping and epoch > afrl.n_encode_epochs:
            train_comb = True
        for i, (x, _, s, _) in enumerate(settings.train_loader):
            # Move batch to GPU if Csr is used
            if settings.csr:
                x = x.to(device=device)
                s = s.to(device=device, dtype=torch.float32)

            # Get latent reps using underlying model in eval mode
            model.eval()
            latent, _, _ = model(x, s, {}, decode=False)

            # AFRL encodings with random mask for each training batch. Switch back to train mode
            mask = np.random.randint(2, size=model.c.n_sensitive)
            model.train()
            combined, feature_encodings = info_align_encode(model, s, latent, device, mask)

            # Combine AFRL encodings into final latents
            final_latents = model.combine_net(combined.detach())

            # Switch back to eval mode and decode the final latent using the underlying model
            model.eval()
            logits, loss = model.decode(x, final_latents, return_logits=combine_bpr)

            # Switch back to train in case adversarial models are applied when updating the model
            model.train()

            # UPDATE AFRL
            # NB: The original paper states that the method updates the parts using the EM algorithm. However,
            # their implementation define each optimizers like we do such that the encoding parts and the
            # combination parts have independent parameters. Thus, we chose to first train the encodings, followed
            # by the combination network
            if not train_comb:
                target_loss, adv_loss = update_encoders(
                    model,
                    settings,
                    latent,
                    s,
                    feature_encodings,
                    afrl.beta,
                    encode_optim,
                    class_optim,
                    neut_encode_optim,
                    adv_optim,
                )
                # epoch_loss.append((target_loss - afrl.lambd * adv_loss).detach().cpu().numpy())
            # Train combination network
            else:
                loss = update_combination_network(
                    x, latent, final_latents, logits, loss, combine_optim, combine_bpr, combine_mse, device
                )
                # epoch_loss.append(loss.detach().cpu().numpy())
            # Debug
            if settings.debug and epoch % 10 == 0 and i == 0:
                debug_afrl(settings, epoch, afrl.n_encode_epochs, x_val, s_val, y_val, loss, target_loss, adv_loss)

        model.eval()
        for i, (x, _, s, _) in enumerate(settings.val_loader):
            # Move batch to GPU if Csr is used
            if settings.csr:
                x = x.to(device=device)
                s = s.to(device=device, dtype=torch.float32)

            # Get latent reps using underlying model in eval mode

            latent, _, _ = model(x, s, {}, decode=False)

            mask = np.ones(model.c.n_sensitive)
            combined, feature_encodings = info_align_encode(model, s, latent, device, mask)

            final_latents = model.combine_net(combined.detach())

            logits, loss = model.decode(x, final_latents, return_logits=combine_bpr)

            if not train_comb:
                target_loss, adv_loss = update_encoders(
                    model,
                    settings,
                    latent,
                    s,
                    feature_encodings,
                    afrl.beta,
                    encode_optim,
                    class_optim,
                    neut_encode_optim,
                    adv_optim,
                )
                epoch_loss = target_loss.detach().cpu().numpy()
            # Train combination network
            else:
                loss = update_combination_network(
                    x, latent, final_latents, logits, loss, combine_optim, combine_bpr, combine_mse, device
                )
                epoch_loss = loss.detach().cpu().numpy()

        # print(f"{epoch_loss}")
        current_n_es = n_es
        if not train_comb:
            current_n_es = 2 * n_es
        if epoch < min_epochs:
            stop = False
        else:
            stop, es_counter, best_loss, best_model = update_early_stopping(
                epoch_loss, best_loss, es_counter, model, best_model, current_n_es, eps=eps
            )
        if stop:
            model.load_state_dict(best_model)
            if train_comb:
                break
            train_comb = True
            best_loss = np.inf
            es_counter = 0

        epoch += 1

    model.add_noise = False
    model.sample = False
