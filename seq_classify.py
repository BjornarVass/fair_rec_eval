import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from item2vec import fit_item2vec


class SeqClassifier(nn.Module):

    def __init__(self, embedding_dim, n_head, seq_len, vocab_size=None):
        super(SeqClassifier, self).__init__()
        self.embed = False
        if vocab_size is not None:
            self.embed = True
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.t1 = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        # self.t2 = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        self.linear1 = nn.Linear(embedding_dim * seq_len, 1)
        # self.linear2 = nn.Linear(16, 1)

    def forward(self, inputs):
        if self.embed:
            inputs = self.embeddings(inputs)
        out = self.t1(inputs)
        # out = self.t2(F.relu(out))
        out = out.reshape(inputs.shape[0], -1)
        out = self.linear1(out)
        # out = self.linear2(F.relu(out))
        out = F.sigmoid(out)
        return out


class SeqDataset(Dataset):
    def __init__(self, seqs, s):
        self.seqs = seqs.detach()
        self.s = torch.tensor(s, dtype=torch.float32)
        self.n = self.seqs.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.seqs[idx], self.s[idx]


def main():
    device = "cuda"
    torch.set_default_device(device)
    min_count = 5
    window_size = 2
    top_k = 10
    embed_top_k = 20
    emb_tr_batch = 64
    embedding_dim = 8
    dim2 = 128
    emb_n_epochs = 25
    store = True
    load = True

    tr_batch = 1024
    n_head = 4
    n_tests = 10
    new_embed = True
    if new_embed:
        n_epochs = [40, 100, 10]
    else:
        n_epochs = [15, 60, 10]

    data = np.load("metrics2.npz")
    tr_x = data["tr_x"]
    tr_s = data["tr_s"]
    tr_p = data["tr_p"]

    va_x = data["va_x"]
    va_s = data["va_s"]
    va_p = data["va_p"]

    if not load:
        embeddings, reindex_map = fit_item2vec(
            tr_x,
            tr_p,
            va_p,
            min_count,
            window_size,
            embed_top_k,
            emb_tr_batch,
            embedding_dim,
            dim2,
            emb_n_epochs,
            device,
            store=store,
        )
    else:
        with open("reindex_map.pkl", "rb") as f:
            reindex_map = pickle.load(f)
        embeddings = nn.Embedding(len(reindex_map), embedding_dim)
        embeddings.load_state_dict(torch.load("embs.pt", weights_only=True))

    vocab_size = None
    if new_embed:
        vocab_size = len(reindex_map)
    tr_pop, tr_rand = fair_baseline_scores(tr_x)
    va_pop, va_rand = fair_baseline_scores(va_x, train_pop=tr_x.sum(0))
    models = ["VAE", "POP", "RAND"]
    for i, mode in enumerate(models):
        if mode == "VAE":
            tr_scores = tr_p
            va_scores = va_p
        elif mode == "POP":
            tr_scores = tr_pop
            va_scores = va_pop
        elif mode == "RAND":
            tr_scores = tr_rand
            va_scores = va_rand
        else:
            exit(1)

        tr_sq_x = get_seq_data(tr_scores, top_k, reindex_map)
        va_sq_x = get_seq_data(va_scores, top_k, reindex_map)

        if not new_embed:
            tr_sq_x = embeddings(tr_sq_x)
            va_sq_x = embeddings(va_sq_x)

        tr_loader = DataLoader(
            SeqDataset(tr_sq_x, tr_s[:, 0:1]),
            batch_size=tr_batch,
            shuffle=True,
            generator=torch.Generator(device=device),
        )
        va_loader = DataLoader(
            SeqDataset(va_sq_x, va_s[:, 0:1]),
            batch_size=va_scores.shape[0],
            generator=torch.Generator(device=device),
        )
        aucs = []
        for test_nr in range(n_tests):
            model = SeqClassifier(embedding_dim, n_head, top_k, vocab_size=vocab_size)
            optimizer = optim.Adam(model.parameters())
            loss_function = nn.BCELoss()
            for epoch in range(n_epochs[i]):
                model.train()
                losses = []
                for x, s in tr_loader:
                    log_probs = model(x)
                    optimizer.zero_grad()
                    loss = loss_function(log_probs, s)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                model.eval()
                with torch.no_grad():
                    va_losses = []
                    auc = []
                    for x, s in va_loader:
                        probs = model(x)
                        loss = loss_function(probs, s)
                        va_losses.append(loss.item())

                        # AUC
                        probs_np = probs.detach().cpu().numpy()
                        s_np = s.detach().cpu().numpy()
                        auc.append(roc_auc_score(s_np, probs_np))
                # print(
                #    f"{epoch}/{n_epochs[i]} Train: {np.mean(losses):.4f}, Val: {np.mean(va_losses):.4f}, AUC: {np.mean(auc)}"
                # )
            aucs.append(roc_auc_score(s_np, probs_np))
        print(f"MODEL: {mode}, AUC: {np.mean(aucs):.4f} std: {np.std(aucs):.4f}")


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


if __name__ == "__main__":
    main()
