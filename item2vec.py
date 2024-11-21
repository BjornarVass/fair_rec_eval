import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dim2):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, dim2)
        self.linear2 = nn.Linear(dim2, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = F.selu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data[:, 0], dtype=torch.int64)
        self.y = torch.tensor(data[:, 1], dtype=torch.int64)
        self.n = self.x.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def fit_item2vec(
    tr_x, tr_p, va_p, min_count, window_size, top_k, tr_batch, embedding_dim, dim2, n_epochs, device, store=False
):
    n_u = tr_x.shape[0]
    n_i = tr_x.shape[1]

    top_recs = np.argsort(-tr_p, axis=1)[:, :top_k]
    unique_recs, counts = np.unique(top_recs, return_counts=True)
    print_rec_dist(n_u, n_i, counts)

    reindex_map = {}
    n_rec_i = 0
    for i, old_ind in enumerate(unique_recs):
        if counts[i] >= min_count:
            reindex_map[old_ind] = n_rec_i
            n_rec_i += 1

    tr_data = item2vec_data(tr_p, top_k, reindex_map, window_size)
    va_data = item2vec_data(va_p, top_k, reindex_map, window_size)
    tr_loader = DataLoader(
        SkipGramDataset(tr_data), batch_size=tr_batch, shuffle=True, generator=torch.Generator(device=device)
    )
    va_loader = DataLoader(
        SkipGramDataset(va_data), batch_size=va_data.shape[0] // 20, generator=torch.Generator(device=device)
    )

    model = SkipGram(n_rec_i, embedding_dim, dim2)
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.NLLLoss()
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for x, y in tr_loader:
            log_probs = model(x)
            optimizer.zero_grad()
            loss = loss_function(log_probs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            va_losses = []
            for x, y in va_loader:
                log_probs = model(x)
                loss = loss_function(log_probs, y)
                va_losses.append(loss.item())
        # print(f"{epoch}/{n_epochs} Train: {np.mean(losses):.4f}, Val: {np.mean(va_losses):.4f}")

    if store:
        torch.save(model.embeddings.state_dict(), "embs.pt")
        with open("reindex_map.pkl", "wb") as f:
            pickle.dump(reindex_map, f)

    return model.embeddings, reindex_map


def item2vec_data(probs, top_k, reindex_map, window_size):
    top_recs = np.argsort(-probs, 1)[:, :10]
    training_data = []
    incs = list(range(-window_size, window_size))
    incs = [x + 1 if x >= 0 else x for x in incs]
    for user_ind in range(top_recs.shape[0]):
        recs = top_recs[user_ind]
        valid_recs = []
        for rec in recs:
            if rec in reindex_map:
                valid_recs.append(reindex_map[rec])

        n_valid = len(valid_recs)
        if n_valid <= 1:
            continue

        for i in range(n_valid):
            for inc in incs:
                inc_ind = i + inc
                if inc_ind >= 0 and inc_ind < n_valid:
                    training_data.append([valid_recs[inc_ind], valid_recs[i]])
    return np.array(training_data, dtype=np.int64)


def print_rec_dist(n_u, n_i, counts):
    n_ones = 0
    for val in counts:
        if val == 1:
            n_ones += 1
    print(f"#Users:       {n_u}")
    print(f"#Items:       {n_i}")
    print(f"#I-rec:       {counts.shape[0]}")
    print(f"#Ones:        {n_ones}")
    print(f"50%:          {np.percentile(counts, 50):.2f}")
    print(f"75%:          {np.percentile(counts, 75):.2f}")
    print(f"90%:          {np.percentile(counts, 90):.2f}")
    print(f"95%:          {np.percentile(counts, 95):.2f}")
    print(f"97%:          {np.percentile(counts, 97):.2f}")
    print(f"98%:          {np.percentile(counts, 98):.2f}")
    print(f"99%:          {np.percentile(counts, 99):.2f}")
    print(f"99.5%:        {np.percentile(counts, 99.5):.2f}")
    print(f"99.8%:        {np.percentile(counts, 99.8):.2f}")
    print(f"99.9%:        {np.percentile(counts, 99.9):.2f}")
    print(f"99.95%:       {np.percentile(counts, 99.95):.2f}")
    print(f"Max:          {np.max(counts)}")
