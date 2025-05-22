import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from item2vec import fit_item2vec
from utils import ndcg_at_k, item_wise_ratio, item_ratio_diff, kendall_tau_rec
from optin.train import parse_arguments
from optin.data_processing import load_and_uncompress


class LinearClassifier(nn.Module):

    def __init__(self, embedding_dim, seq_len, vocab_size=None, cat_ratio=False):
        super(LinearClassifier, self).__init__()
        self.cat_ratio = cat_ratio
        self.embed = False
        if vocab_size is not None:
            self.embed = True
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        hidden = 32
        hidden_dim = embedding_dim
        if self.cat_ratio:
            hidden_dim += 1
        self.linear1 = nn.Linear(hidden_dim * seq_len, hidden)
        # self.linear2 = nn.Linear(hidden, hidden)
        self.linear_last = nn.Linear(hidden, 1)

    def forward(self, inputs, ratios):
        if self.embed:
            inputs = self.embeddings(inputs)
        # TODO REMOVE
        # inputs = torch.zeros(inputs.shape)
        if self.cat_ratio:
            inputs = torch.cat((inputs, ratios), 2)
            # inputs = 1 - ratios
        inputs = inputs.reshape((inputs.shape[0]), -1)
        out = self.linear1(inputs)
        # out = self.linear2(F.selu(out))
        out = self.linear_last(F.selu(out))
        out = F.sigmoid(out)
        return out


class RNNClassifier(nn.Module):

    def __init__(self, embedding_dim, seq_len, vocab_size=None, cat_ratio=False):
        super(RNNClassifier, self).__init__()
        self.cat_ratio = cat_ratio
        self.embed = False
        if vocab_size is not None:
            self.embed = True
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        hidden = 32
        self.bidirectional = True
        rnn_dim = embedding_dim
        if self.cat_ratio:
            rnn_dim += 1
        self.rnn = nn.LSTM(rnn_dim, hidden, batch_first=True, bidirectional=self.bidirectional)
        # self.linear2 = nn.Linear(hidden, hidden)
        self.linear_last = nn.Linear(2 * hidden, 1)

    def forward(self, inputs, ratios):
        if self.embed:
            inputs = self.embeddings(inputs)
        if self.cat_ratio:
            inputs = torch.cat((inputs, ratios), 2)
        _, out = self.rnn(inputs)
        if self.bidirectional:
            out = torch.concat((out[0][0], out[0][1]), dim=1)
        else:
            out = out[0].reshape(inputs.shape[0], -1)
        # out = self.linear2(F.selu(out))
        out = self.linear_last(out)
        out = F.sigmoid(out)
        return out


class SeqClassifier(nn.Module):

    def __init__(self, embedding_dim, n_head, seq_len, vocab_size=None, cat_ratio=False):
        super(SeqClassifier, self).__init__()
        self.cat_ratio = cat_ratio
        self.embed = False
        if vocab_size is not None:
            self.embed = True
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        hidden_dim = embedding_dim
        if self.cat_ratio:
            hidden_dim += 1
        self.t1 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, batch_first=True, activation="gelu")
        # self.t2 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * seq_len, 1)
        # self.linear2 = nn.Linear(16, 1)

    def forward(self, inputs, ratios):
        if self.embed:
            inputs = self.embeddings(inputs)
        # TODO REMOVE
        # inputs = torch.zeros(inputs.shape)
        if self.cat_ratio:
            inputs = torch.cat((inputs, ratios), 2)
        out = self.t1(inputs)
        # out = self.t2(F.relu(out))
        out = out.reshape(inputs.shape[0], -1)
        out = self.linear1(out)
        # out = self.linear2(F.relu(out))
        out = F.sigmoid(out)
        return out


class SeqDataset(Dataset):
    def __init__(self, seqs, s, dropout=False, ratios=None, device="cuda"):
        self.dropout = dropout
        self.seqs = seqs.detach()
        self.s = torch.tensor(s, dtype=torch.float32, device=device)
        self.n = self.seqs.shape[0]
        self.ratios = ratios

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ratios = 0
        if self.dropout:
            n = self.seqs.shape[1]
            all_idx = np.arange(n)
            mask = np.random.rand(n // 2) < 0.3
            all_idx[mask.nonzero()] = -1
            out_inds = np.zeros(n // 2)

            next_idx = 0
            for val in all_idx:
                if val != -1:
                    out_inds[next_idx] = val
                    next_idx += 1
                    if next_idx == n // 2:
                        break
            seq = seq[out_inds]
            if self.ratios is not None:
                ratios = self.ratios[idx, out_inds]
        elif self.ratios is not None:
            ratios = self.ratios[idx]
        return seq, self.s[idx], ratios


def main():
    dataset_path = "synth.npz"
    # data = "metrics_afrl.npz"
    # data = "metrics_perturb2adv.npz"
    eval_classify(dataset_path)


def eval_classify(
    dataset_path,
    dataset_filename,
    budget=50,
    info_lambd=5.0,
    info_adv=10,
    embeddings=None,
    synth=True,
    s_ind=0,
    double_adv=False,
):
    device = "cuda"
    # torch.set_default_device(device)
    top_k = 40
    small_k = 10
    embedding_dim = 8

    cat_ratio = True
    new_embed = False
    all_trained_models = False
    perturb = budget > -1
    info_align = info_lambd > -1
    train_rec = perturb or info_align
    models = ["POP ", "RAND", "UNFAIR", "UNFAIR_DIV"]
    if train_rec:
        models = models + ["BASE"]
        if perturb:
            models = models + ["PERTURB"]
        if info_align:
            models = models + ["INFO"]

    tr_batch = 1024
    n_head = 2 + cat_ratio
    n_es = 5
    n_tests = 1
    verbose = False
    log_reg = False
    dropout = False
    if synth:
        data = np.load(dataset_path + dataset_filename)
        # data = np.load("metrics_afrl.npz")
        # data = np.load("metrics_perturb2adv.npz")
        tr_x = data["tr_x"]
        tr_s = data["tr_s"]

        va_x = data["va_x"]
        va_s = data["va_s"]
        va_y = data["va_y"]

        te_x = data["te_x"]
        te_s = data["te_s"]
        te_y = data["te_y"]
    else:
        (
            tr_x,
            va_x,
            va_y,
            te_x,
            te_y,
            _,
            tr_s,
            va_s,
            te_s,
            _,
        ) = load_and_uncompress(dataset_path, user_split=True)

    if all_trained_models:
        models = ["VAE ", "FAIR"] + models
        tr_p = data["tr_p"]
        tr_p2 = data["tr_p2"]
        va_p = data["va_p"]
        va_p2 = data["va_p2"]
        te_p = data["te_p"]
        te_p2 = data["te_p2"]
    elif train_rec:
        if perturb:
            tr_ba, va_ba, te_ba, tr_perturb, va_perturb, te_perturb, auc_base, auc_perturb = train_models(
                dataset_path, perturb=True, budget=budget, synth=synth
            )
        if info_align:
            tr_ba, va_ba, te_ba, tr_info, va_info, te_info, auc_base, auc_info = train_models(
                dataset_path,
                info_align=True,
                lambd=info_lambd,
                n_adv_train=info_adv,
                synth=synth,
                double_adv=double_adv,
            )
    demographics = tr_s.mean(0)

    if new_embed:
        reindex_map = {i: i for i in range(tr_x.shape[1])}
    elif embeddings is not None:
        embeddings, reindex_map = embeddings

    base_item_ratios = tr_x[tr_s[:, s_ind] == 1].sum(0) / tr_x.sum(0)
    item_ratios = None
    if cat_ratio:
        item_ratios = np.zeros(len(reindex_map), dtype=np.float32)
        for k, v in reindex_map.items():
            ratio = base_item_ratios[k]
            item_ratios[v] = ratio
        item_ratios = torch.tensor(item_ratios, device=device)

    vocab_size = None
    if new_embed:
        vocab_size = len(reindex_map)
    tr_pop, tr_rand = fair_baseline_scores(tr_x)
    va_pop, va_rand = fair_baseline_scores(va_x, train_pop=tr_x.sum(0))
    te_pop, te_rand = fair_baseline_scores(te_x, train_pop=tr_x.sum(0))

    tr_unfair, tr_divisive = unfair_baseline_scores(tr_x, tr_s[:, s_ind])
    va_unfair, va_divisive = unfair_baseline_scores(va_x, va_s[:, s_ind])
    te_unfair, te_divisive = unfair_baseline_scores(te_x, te_s[:, s_ind])

    results = {}

    for i, mode in enumerate(models):
        if mode == "VAE ":
            tr_scores = tr_p
            va_scores = va_p
            te_scores = te_p
        elif mode == "FAIR":
            tr_scores = tr_p2
            va_scores = va_p2
            te_scores = te_p2
        elif mode == "POP ":
            tr_scores = tr_pop
            va_scores = va_pop
            te_scores = te_pop
        elif mode == "RAND":
            tr_scores = tr_rand
            va_scores = va_rand
            te_scores = te_rand
        elif mode == "UNFAIR":
            tr_scores = tr_unfair
            va_scores = va_unfair
            te_scores = te_unfair
        elif mode == "UNFAIR_DIV":
            tr_scores = tr_divisive
            va_scores = va_divisive
            te_scores = te_divisive
        elif mode == "BASE":
            tr_scores = tr_ba
            va_scores = va_ba
            te_scores = te_ba
        elif mode == "PERTURB":
            tr_scores = tr_perturb
            va_scores = va_perturb
            te_scores = te_perturb
        elif mode == "INFO":
            tr_scores = tr_info
            va_scores = va_info
            te_scores = te_info
        else:
            continue

        train_seq_len = top_k if log_reg or not dropout else 2 * top_k
        tr_sq_x = get_seq_data(tr_scores, train_seq_len, reindex_map)
        va_sq_x = get_seq_data(va_scores, top_k, reindex_map)
        te_sq_x = get_seq_data(te_scores, top_k, reindex_map)

        tr_ratios = None
        va_ratios = None
        te_ratios = None
        if cat_ratio:
            tr_ratios = torch.unsqueeze(torch.take(item_ratios, tr_sq_x), 2)
            va_ratios = torch.unsqueeze(torch.take(item_ratios, va_sq_x), 2)
            te_ratios = torch.unsqueeze(torch.take(item_ratios, te_sq_x), 2)

        if not new_embed:
            tr_sq_x = embeddings(tr_sq_x)
            va_sq_x = embeddings(va_sq_x)
            te_sq_x = embeddings(te_sq_x)

        if log_reg:
            logreg(tr_sq_x, va_sq_x, tr_s[:, s_ind], va_s[:, s_ind])
            continue

        tr_loader = DataLoader(
            SeqDataset(tr_sq_x, tr_s[:, s_ind : s_ind + 1], dropout=dropout, ratios=tr_ratios),
            batch_size=tr_batch,
            shuffle=True,
        )
        va_loader = DataLoader(
            SeqDataset(va_sq_x, va_s[:, s_ind : s_ind + 1], ratios=va_ratios),
            batch_size=va_scores.shape[0],
        )
        te_loader = DataLoader(
            SeqDataset(te_sq_x, te_s[:, s_ind : s_ind + 1], ratios=te_ratios),
            batch_size=te_scores.shape[0],
        )
        aucs = []
        for test_nr in range(n_tests):
            # model = SeqClassifier(embedding_dim, n_head, top_k, vocab_size=vocab_size, cat_ratio=cat_ratio)
            # model = LinearClassifier(embedding_dim, top_k, vocab_size=vocab_size, cat_ratio=cat_ratio)
            model = RNNClassifier(embedding_dim, top_k, vocab_size=vocab_size, cat_ratio=cat_ratio)
            model.to(device)
            optimizer = optim.Adam(model.parameters())
            loss_function = nn.BCELoss()
            best_loss = np.inf
            improv_counter = 0
            best_model = None
            epoch = 0
            eps = 1e-4
            while epoch < 1000:
                model.train()
                losses = []
                for x, s, ratios in tr_loader:
                    log_probs = model(x, ratios)
                    optimizer.zero_grad()
                    loss = loss_function(log_probs, s)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                model.eval()
                with torch.no_grad():
                    va_losses = []
                    auc = []
                    for x, s, ratios in va_loader:
                        probs = model(x, ratios)
                        loss = loss_function(probs, s)
                        va_losses.append(loss.item())

                        # AUC
                        probs_np = probs.detach().cpu().numpy()
                        s_np = s.detach().cpu().numpy()
                        auc.append(roc_auc_score(s_np, probs_np))
                mean_loss = np.mean(va_losses)
                if verbose:
                    print(f"{epoch} Train: {np.mean(losses):.4f}, Val: {mean_loss:.4f}, AUC: {np.mean(auc)}")
                # Early stopping
                if mean_loss < best_loss:
                    best_model = copy.deepcopy(model.state_dict())
                    if best_loss - mean_loss > eps:
                        improv_counter = 0
                    else:
                        improv_counter += 1
                        if improv_counter == n_es:
                            break
                    best_loss = mean_loss
                else:
                    improv_counter += 1
                    if improv_counter == n_es:
                        break
                epoch += 1

            # Test data
            model.load_state_dict(best_model)
            model.eval()
            with torch.no_grad():
                te_losses = []
                auc = []
                for x, s, ratios in te_loader:
                    probs = model(x, ratios)
                    loss = loss_function(probs, s)
                    te_losses.append(loss.item())

                    # AUC
                    probs_np = probs.detach().cpu().numpy()
                    s_np = s.detach().cpu().numpy()
                    auc.append(roc_auc_score(s_np, probs_np))
            aucs.append(np.mean(auc))

        rat_auc_mean, rat_auc_med = ratio_auc(te_scores, s_np, top_k, base_item_ratios)
        met_item_ratios, item_counts = item_wise_ratio(te_scores, te_s, top_k)
        ratio_mean, ratio_std = item_ratio_diff(met_item_ratios, demographics, item_counts, lower_count=5)
        kendall_tau = kendall_tau_rec(te_x, s_np[:, 0], top_k, 2 * top_k, te_scores)

        rat_auc_mean2, rat_auc_med2 = ratio_auc(te_scores, s_np, small_k, base_item_ratios)
        met_item_ratios2, item_counts2 = item_wise_ratio(te_scores, te_s, small_k)
        ratio_mean2, ratio_std2 = item_ratio_diff(met_item_ratios2, demographics, item_counts2, lower_count=5)
        kendall_tau2 = kendall_tau_rec(te_x, s_np[:, 0], small_k, 2 * small_k, te_scores)
        # print(
        #     f"MODEL: {mode}, AUC: {np.mean(aucs):.4f} std: {np.std(aucs):.4f} R_Mean_AUC: {rat_auc_mean:.4f} R_Med_AUC: {rat_auc_med:.4f} NDCG: {np.mean(ndcg_at_k(te_scores, te_y)):.4f} Item ratio: {ratio_mean[s_ind]:.4f}"
        # )
        model_results = {
            "AUC": np.mean(aucs),
            "AUC_std": np.std(aucs),
            "R_Mean_AUC": rat_auc_mean,
            "R_Med_AUC": rat_auc_med,
            "NDCG": np.mean(ndcg_at_k(te_scores, te_y)),
            "Item_ratio": ratio_mean[s_ind],
            "Kendall-Tau": kendall_tau,
            f"R_Mean_AUC{small_k}": rat_auc_mean2,
            f"R_Med_AUC{small_k}": rat_auc_med2,
            f"Item_ratio{small_k}": ratio_mean2[s_ind],
            f"Kendall-Tau{small_k}": kendall_tau2,
        }
        if mode == "BASE":
            model_results["Rep AUC"] = auc_base
        elif mode == "PERTURB":
            model_results["Rep AUC"] = auc_perturb
        elif mode == "INFO":
            model_results["Rep AUC"] = auc_info
        results[mode] = model_results
    return results


def train_models(
    datapath, perturb=False, info_align=False, budget=50, lambd=5.0, n_adv_train=10, synth=True, double_adv=False
):
    args = [
        "--beta",
        "0.2",
        "--processed_dir",
        datapath,
        "--tr_batch_size",
        "805",
        "--te_batch_size",
        "1610",
        "--adv",
        "--early_stopping",
        "25",
    ]
    if synth:
        args += ["--load_synth"]
    if perturb:
        args += [
            "--perturb",
            "--dropout_adv",
            "0.5",
            "--hidden_dim_adv",
            "32",
            "--max_budget",
            str(budget),
        ]
        if double_adv:
            args += ["--double_adv"]
    elif info_align:
        args += [
            "--info_align",
            "--hidden_dim_adv",
            "64",
            "--n_adv_pre",
            "0",
            "--lambd",
            str(lambd),
            "--n_adv_train",
            str(n_adv_train),
        ]
    base_probs, fair_probs, base_auc, fair_auc = parse_arguments(args)
    tr_p = base_probs[0]
    va_p = base_probs[1]
    te_p = base_probs[2]
    tr_q = fair_probs[0]
    va_q = fair_probs[1]
    te_q = fair_probs[2]
    return tr_p, va_p, te_p, tr_q, va_q, te_q, base_auc, fair_auc


def get_seq_data(probs, top_k, reindex_map, tensor=True, device="cuda"):
    n_users = probs.shape[0]
    n_items = probs.shape[1]
    valid_recs = np.zeros((n_users, top_k), dtype=np.int64)
    for i in range(n_users):
        sorted_args = np.argsort(-probs[i])
        next_ind = 0
        for j in range(n_items):
            cand_arg = sorted_args[j]
            if cand_arg in reindex_map:
                valid_recs[i, next_ind] = reindex_map[cand_arg]
                next_ind += 1
                if next_ind == top_k:
                    break

    if tensor:
        valid_recs = torch.tensor(valid_recs, device=device)
    return valid_recs


def logreg(tr_seqs, va_seqs, tr_s, va_s):
    tr_seqs = tr_seqs.detach().cpu().numpy()
    tr_seqs = tr_seqs.reshape(tr_seqs.shape[0], -1)
    va_seqs = va_seqs.detach().cpu().numpy()
    va_seqs = va_seqs.reshape(va_seqs.shape[0], -1)

    model = LogisticRegression(C=0.005).fit(tr_seqs, tr_s)
    tr_probs = model.predict_proba(tr_seqs)
    va_probs = model.predict_proba(va_seqs)
    tr_auc = roc_auc_score(tr_s, tr_probs[:, 1])
    va_auc = roc_auc_score(va_s, va_probs[:, 1])
    print(f"TR: {tr_auc} VA: {va_auc}")


def fair_baseline_scores(x, train_pop=None):
    # Popularity is based on train set. Has to be provided for val and test set
    if train_pop is None:
        train_pop = x.sum(0)
    pop_scores = np.tile(train_pop, (x.shape[0], 1))
    pop_scores[x.nonzero()] = -np.inf

    # Each row sorted independently, large to small values
    random_scores = np.argsort(-np.random.rand(*x.shape), axis=1).astype(np.float32)
    random_scores[x.nonzero()] = -np.inf
    return pop_scores, random_scores


def unfair_baseline_scores(x, s):
    s_pop = np.zeros_like(x)
    s_divisive = np.zeros_like(x)
    pop0 = x[s == 0]
    pop1 = x[s == 1]
    n0 = pop0.shape[0]
    n1 = pop1.shape[0]
    sum0 = pop0.sum(0)
    sum1 = pop1.sum(0)
    s_pop[s == 0] = sum0
    s_pop[s == 1] = sum1
    s_pop[x.nonzero()] = -np.inf

    tot_count = x.sum(0)
    ratio0 = sum0 / n0
    ratio1 = sum1 / n1
    ratio_score0 = ratio0 - ratio1
    ratio_score1 = ratio1 - ratio0
    ratio_score0[np.logical_or(ratio_score0 < 0, tot_count < 5)] = 0
    ratio_score1[np.logical_or(ratio_score1 < 0, tot_count < 5)] = 0
    s_divisive[s == 0] = ratio_score0
    s_divisive[s == 1] = ratio_score1
    s_divisive[x.nonzero()] = -np.inf

    return s_pop, s_divisive


def ratio_auc(probs, s, top_k, ratios):
    recs = np.argsort(-probs)[:, :top_k]
    ratios = np.take(ratios, recs)
    mean_ratio = ratios.mean(1)
    median_ratio = np.median(ratios, 1)
    return roc_auc_score(s, mean_ratio), roc_auc_score(s, median_ratio)


if __name__ == "__main__":
    main()
