import numpy as np
import copy
from optin.model import EvalResults
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def generate_network_dims(input_dim, hidden_dim, output_dim, n_hidden=0, red_hidden_dim=-1):
    network_dims = [input_dim]
    for i in range(n_hidden):
        if i == 0 or red_hidden_dim < 0:
            network_dims.append(hidden_dim)
        else:
            network_dims.append(red_hidden_dim)
    network_dims.append(output_dim)
    return network_dims


def update_early_stopping(loss, best_loss, es_counter, model, best_model, n_es, eps=0.0):
    check_eps = eps > 0
    if loss < best_loss:
        if not check_eps or best_loss - loss > eps:
            es_counter = 0
        elif check_eps and best_loss - loss <= eps:
            es_counter += 1
        best_loss = loss
        best_model = copy.deepcopy(model.state_dict())
    else:
        es_counter += 1

    stop = es_counter == n_es
    return stop, es_counter, best_loss, best_model


def get_ranking_info(probs, rec_targets, k, rec_indexes):
    # Disregard users with no correct recommendations
    mask = rec_targets.sum(axis=1) != 0
    targets = rec_targets[mask]

    n_users = targets.shape[0]

    if rec_indexes is None:
        probs = probs[mask]

        # Partition to place indexes of k top probs (unsorted) in the begining of the list
        index_partition = np.argpartition(-probs, k, axis=1)

        # Top k probs (unsorted)
        topk_probs = probs[np.arange(n_users)[:, np.newaxis], index_partition[:, :k]]
        # Top k internal indexes (sorted, indexed 0-(k-1))
        topk_sorted_indexes = np.argsort(-topk_probs, axis=1)

        # Top k SORTED ITEM indexes
        rec_indexes = index_partition[np.arange(n_users)[:, np.newaxis], topk_sorted_indexes]
    else:
        rec_indexes = rec_indexes[mask]

    return rec_indexes, targets, n_users


def ndcg_at_k(probs, rec_targets, k=10, rec_indexes=None):
    rec_indexes, targets, n_users = get_ranking_info(probs, rec_targets, k, rec_indexes)

    # Rank discount
    rank_discount = 1.0 / np.log2(np.arange(2, k + 2))

    # DCG utilize that targets are either 1 or 0 and multiplies with the rank discounts
    # IDCG simply assumes that min(k, n_targets) targets inhabits the best rankings
    DCG = (targets[np.arange(n_users)[:, np.newaxis], rec_indexes] * rank_discount).sum(axis=1)
    IDCG = np.array([(rank_discount[: min(int(n), k)]).sum() for n in targets.sum(axis=1)])

    NDCG = DCG / IDCG
    return NDCG


def item_wise_ratio(scores, s, k, ranked=False):
    n_s = s.shape[1]
    n_items = scores.shape[1]
    ratios = np.zeros((n_s, n_items))
    counts = None
    ranked_discount = 1.0 / np.log2(np.arange(2, k + 2))
    for i in range(n_s):
        counters = np.zeros((2, n_items))
        for sens in [0, 1]:
            mask = s[:, i] == sens
            top_k = np.argsort(-scores[mask], axis=1)[:, :k]
            for j in range(top_k.shape[0]):
                for l, ind in enumerate(top_k[j]):
                    if ranked:
                        counters[sens, ind] += ranked_discount[l]
                    else:
                        counters[sens, ind] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = counters[1] / counters.sum(0)
        ratios[i] = ratio
        counts = counters.sum(0)
    return ratios, counts


def user_item_scores(scores, k, ratios, ratio_counts, lower_count, ranked=False):
    recs = np.argsort(-scores, axis=1)[:, :k]
    scores = np.zeros((recs.shape[0], ratios.shape[0]))
    counts = np.zeros(recs.shape[0])
    ranked_discount = 1.0 / np.log2(np.arange(2, k + 2))
    for i in range(recs.shape[0]):
        for l, rec in enumerate(recs[i]):
            if ratio_counts[rec] >= lower_count:
                counts[i] += 1
                for j in range(ratios.shape[0]):
                    if ranked:
                        scores[i, j] += ranked_discount[l]
                    else:
                        scores[i, j] += ratios[j, rec]
    return scores, counts


def dump_ratio(scores, s, k, demographics, file_path, lower_count, ranked, old_scores):
    ratios, counts = item_wise_ratio(scores, s, k, ranked=ranked)
    if old_scores is not None:
        old_ratios, old_counts = item_wise_ratio(old_scores, s, k, ranked=ranked)
    user_scores, user_counts = user_item_scores(scores, k, ratios, counts, lower_count, ranked=ranked)
    if old_scores is not None:
        old_user_scores, old_user_counts = user_item_scores(
            old_scores, k, old_ratios, old_counts, lower_count, ranked=ranked
        )
    ranked_suffix = "_rank" if ranked else ""
    fp = file_path + f"/{k}_{lower_count}ratios{ranked_suffix}.npz"
    if old_scores is not None:
        np.savez(
            fp,
            ratios=ratios,
            counts=counts,
            user_scores=user_scores,
            user_counts=user_counts,
            demographics=demographics,
            s=s,
            old_ratios=old_ratios,
            old_counts=old_counts,
            old_user_scores=old_user_scores,
            old_user_counts=old_user_counts,
        )
    else:
        np.savez(
            fp,
            ratios=ratios,
            counts=counts,
            user_scores=user_scores,
            user_counts=user_counts,
            demographics=demographics,
            s=s,
        )
    return


def dump_torch_ratio(scores, s, k, demographics, file_path, lower_count=5, ranked=False, old_scores=None):
    scores = scores.detach().cpu().numpy()
    s = s.detach().cpu().numpy()
    if old_scores is not None:
        old_scores = old_scores.detach().cpu().numpy()
    dump_ratio(scores, s, k, demographics, file_path, lower_count, ranked, old_scores)


def item_ratio_diff(ratios, demographics, counts, lower_count=-1):
    n_s = demographics.shape[0]
    diffs = np.abs(ratios - demographics[:, np.newaxis])
    means = []
    stds = []
    for i in range(n_s):
        valid_diffs = diffs[i][np.logical_and(~np.isnan(diffs[i]), counts >= lower_count)]
        means.append(np.mean(valid_diffs))
        stds.append(np.std(valid_diffs))
    return means, stds


# Old function. Not used but inspired User ratio plots found in jupyter notebook
def item_ratio_improvement(old_ratios, new_ratios, demographics, counts, lower_count=-1):
    # Hard coded bin size
    bin_size = 0.1
    bins = [0.0, 0.0000000000001, 0.05, 0.1000000001]
    while bins[-2] < 1.0:
        bins.append(bins[-1] + bin_size)
    bins[-2] = 1.0
    bins[-1] = 1.000000001
    bins = np.array(bins)
    n_bins = bins.shape[0]
    n_s = demographics.shape[0]
    all_bin_means = []

    # Filter out items with a lower than minimum rec count
    mask = np.logical_and(
        np.logical_and(~np.isnan(old_ratios.sum(0)), ~np.isnan(new_ratios.sum(0))), counts > lower_count
    )
    old_ratios = old_ratios[:, mask]
    new_ratios = new_ratios[:, mask]
    counts = counts[mask]

    # Calculate diffs and "reverse" diffs where the original ratio is greater than ideal
    diffs = new_ratios - old_ratios
    for i in range(n_s):
        diffs[i, old_ratios[i, :] > demographics[i]] = -diffs[i, old_ratios[i, :] > demographics[i]]

    # For each sensitive attribute
    for i in range(n_s):
        # Tally up the times each bin is covered and the total diffs they make out
        bin_counts = np.zeros(n_bins)
        post_counts = np.zeros(n_bins)
        bin_count_sums = np.zeros(n_bins)
        post_count_sums = np.zeros(n_bins)
        bin_sums = np.zeros(n_bins)
        binned = np.digitize(old_ratios[i, :], bins)
        post_binned = np.digitize(new_ratios[i, :], bins)
        for j in range(diffs.shape[1]):
            current_bin = binned[j]
            bin_counts[current_bin] += 1
            bin_sums[current_bin] += diffs[i, j]

            post_counts[post_binned[j]] += 1

            bin_count_sums[current_bin] += counts[j]
            post_count_sums[post_binned[j]] += counts[j]
        # Calculate mean diffs of each bin
        with np.errstate(divide="ignore", invalid="ignore"):
            bin_means = bin_sums / bin_counts
            bin_count_means = bin_count_sums / bin_counts
            post_count_means = post_count_sums / post_counts
        all_bin_means.append(bin_means)
        for j in range(1, n_bins):
            print(
                f"{bins[j-1]:.2f}-{bins[j]:.2f}: {bin_means[j]:.4f} # {int(bin_counts[j])} Post # {int(post_counts[j])} Avg count {bin_count_means[j]:.1f} Avg post count {post_count_means[j]:.1f}"
            )
    return all_bin_means


def evaluate_all_recommendations(settings, u_probs, targets):
    ndcg = ndcg_at_k(u_probs, targets, settings.k).mean()

    eval_res = EvalResults(settings.model.c.n_sensitive)
    eval_res.set_results(ndcg)
    return eval_res


def evaluate_representation(z, s, mode_str, model, wandb_log):
    # Evaluate how well sensitive features can be inferred from latent representation
    # Fit and evaluate auxiliary model
    n_redundancy = 20
    k_split = 5
    logreg = logreg_training(n_redundancy, k_split, z, s)
    for i, sensitive_name in enumerate(model.c.sensitive_labels):
        wandb_log[f"analysis/{sensitive_name} rep logreg, {mode_str}"] = logreg[i]
    return logreg


def evaluate_filter_representation(data_loader, model):
    n_redundancy = 4
    k_split = 5
    x = data_loader.dataset.x
    s = data_loader.dataset.s
    s_np = s.detach().cpu().numpy()
    model.eval()
    for mask in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        mask = np.array(mask)
        z, _, _ = model(x, s, {}, decode=False, mask=mask)
        z_np = z.detach().cpu().numpy()
        log_reg = logreg_training(n_redundancy, k_split, z_np, s_np)
        print(mask)
        print(log_reg)


def logreg_training(n_redundancy, k_split, z, s, zp=None, frac=0.0):
    if zp is not None and frac > 0.0:
        n = len(zp)
        cutoffs = (((np.arange(n + 1)) * frac) * zp[0].shape[0]).astype(np.int64)
        opt_in = np.arange(z.shape[0])
        np.random.shuffle(opt_in)
        # Simulate users opting for fairness switching a fraction of the
        # representations out with perturbed representations. For frac=0.2
        # and two binary sensitive attributes, a total of 60% of reps will be
        # perturbed (0,1 1,0 1,1). The first "opt" group is used for testing
        for i in range(n):
            l = cutoffs[i]
            u = cutoffs[i + 1]
            inds = opt_in[l:u]
            z[inds] = zp[i][inds]
        z_val = zp[0]
    else:
        z_val = z
    n_sensitive = s.shape[1]
    logreg = [[] for i in range(n_sensitive)]
    for i in range(n_redundancy):
        for j in range(n_sensitive):
            y = s[:, j]

            # Perform random split of data
            kf = KFold(n_splits=k_split, shuffle=True)

            for train_index, val_index in kf.split(z):
                train_z, train_y = z[train_index], y[train_index]
                val_z, val_y = z_val[val_index], y[val_index]

                # Fit and evaluate model
                logreg_model = LogisticRegression(max_iter=200)
                logreg_model.fit(train_z, train_y)
                logreg_probs = logreg_model.predict_proba(val_z)
                logreg[j].append(roc_auc_score(val_y, logreg_probs[:, 1]))

    logreg = [np.mean(logre) for logre in logreg]

    return logreg
