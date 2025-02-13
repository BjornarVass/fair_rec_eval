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

from item2vec import fit_item2vec, fit_item2vec2
from utils import ndcg_at_k, item_wise_ratio, item_ratio_diff


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
    def __init__(self, seqs, s, dropout=False, ratios=None):
        self.dropout = dropout
        self.seqs = seqs.detach()
        self.s = torch.tensor(s, dtype=torch.float32)
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


def eval_classify(dataset_path, dataset_filename):
    device = "cuda"
    torch.set_default_device(device)
    min_count = 5
    window_size = 2
    top_k = 40
    embed_top_k = 20
    emb_tr_batch = 64
    embedding_dim = 8
    dim2 = 128
    emb_n_epochs = 25
    store = True

    cat_ratio = False
    new_embed = False
    load = False
    all_models = False
    models = ["POP ", "RAND"]

    tr_batch = 1024
    n_head = 2 + cat_ratio
    n_es = 5
    n_tests = 10
    verbose = False
    log_reg = False
    dropout = False
    s_ind = 0

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

    if all_models:
        models = ["VAE ", "FAIR"] + models
        tr_p = data["tr_p"]
        tr_p2 = data["tr_p2"]
        va_p = data["va_p"]
        va_p2 = data["va_p2"]
        te_p = data["te_p"]
        te_p2 = data["te_p2"]

    demographics = tr_s.mean(0)

    if new_embed:
        reindex_map = {i: i for i in range(tr_x.shape[1])}
    elif not load:
        # embeddings, reindex_map = fit_item2vec(
        #     tr_x,
        #     tr_p,
        #     va_p,
        #     min_count,
        #     window_size,
        #     embed_top_k,
        #     emb_tr_batch,
        #     embedding_dim,
        #     dim2,
        #     emb_n_epochs,
        #     device,
        #     store=store,
        # )
        embeddings, reindex_map = fit_item2vec2(
            tr_x,
            min_count,
            emb_tr_batch,
            embedding_dim,
            dim2,
            device,
            dataset_path,
            verbose,
            store=store,
        )
    else:
        with open(dataset_path + "reindex_map.pkl", "rb") as f:
            reindex_map = pickle.load(f)
        embeddings = nn.Embedding(len(reindex_map), embedding_dim)
        embeddings.load_state_dict(torch.load("embs.pt", weights_only=True))

    base_item_ratios = tr_x[tr_s[:, s_ind] == 1].sum(0) / tr_x.sum(0)
    item_ratios = None
    if cat_ratio:
        item_ratios = np.zeros(len(reindex_map), dtype=np.float32)
        for k, v in reindex_map.items():
            ratio = base_item_ratios[k]
            item_ratios[v] = ratio
        item_ratios = torch.tensor(item_ratios)

    vocab_size = None
    if new_embed:
        vocab_size = len(reindex_map)
    tr_pop, tr_rand = fair_baseline_scores(tr_x)
    va_pop, va_rand = fair_baseline_scores(va_x, train_pop=tr_x.sum(0))
    te_pop, te_rand = fair_baseline_scores(te_x, train_pop=tr_x.sum(0))

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
            generator=torch.Generator(device=device),
        )
        va_loader = DataLoader(
            SeqDataset(va_sq_x, va_s[:, s_ind : s_ind + 1], ratios=va_ratios),
            batch_size=va_scores.shape[0],
            generator=torch.Generator(device=device),
        )
        te_loader = DataLoader(
            SeqDataset(te_sq_x, te_s[:, s_ind : s_ind + 1], ratios=te_ratios),
            batch_size=te_scores.shape[0],
            generator=torch.Generator(device=device),
        )
        aucs = []
        for test_nr in range(n_tests):
            # model = SeqClassifier(embedding_dim, n_head, top_k, vocab_size=vocab_size, cat_ratio=cat_ratio)
            model = LinearClassifier(embedding_dim, top_k, vocab_size=vocab_size, cat_ratio=cat_ratio)
            # model = RNNClassifier(embedding_dim, top_k, vocab_size=vocab_size, cat_ratio=cat_ratio)
            optimizer = optim.Adam(model.parameters())
            loss_function = nn.BCELoss()
            best_loss = np.inf
            improv_counter = 0
            best_model = None
            epoch = 0
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
                    best_loss = mean_loss
                    improv_counter = 0
                    best_model = copy.deepcopy(model.state_dict())
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
        }
        results[mode] = model_results
    return results


def get_seq_data(probs, top_k, reindex_map, tensor=True):
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

    if tensor == True:
        valid_recs = torch.tensor(valid_recs)
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


def ratio_auc(probs, s, top_k, ratios):
    recs = np.argsort(-probs)[:, :top_k]
    ratios = np.take(ratios, recs)
    mean_ratio = ratios.mean(1)
    median_ratio = np.median(ratios, 1)
    return roc_auc_score(s, mean_ratio), roc_auc_score(s, median_ratio)


if __name__ == "__main__":
    main()
