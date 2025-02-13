import os
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
import powerlaw

import argparse
import json
import sys

RNG_ALTERNATIVE = False
FASTER_RNG = True
VERBOSE = False


def show_interactions_plot(data, saving_path):

    plt.figure()

    data["Score"] = np.ones(data.shape[0])
    data_pivot = data[["Item", "User", "Score"]].pivot(index="Item", columns="User", values="Score")

    plt.imshow(data_pivot, interpolation="nearest", aspect="auto")

    plt.xlabel("Users", fontsize=20)
    plt.ylabel("Items", fontsize=20)
    plt.tight_layout()

    plt.savefig(saving_path + f"intersections.pdf", dpi=200)

    # plt.show()

    plt.clf()


def plot_degree_distributions(distributions, images_folder, category=None, population=None):

    if distributions[0] is not None and distributions[2] is not None:
        plt.figure()

        sns.scatterplot(x=distributions[0], y=distributions[1], label="Items", s=100)
        sns.scatterplot(x=distributions[2], y=distributions[3], label="Users", s=100)

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Degree")

        plt.tight_layout()

        fn = "degree_distributions.pdf"
        image_path = os.path.join(images_folder, fn)

        plt.savefig(image_path)
        # plt.show()

        plt.clf()

    if distributions[0] is not None:

        label = "Items"
        fn = "degree_distributions_top"

        if category is not None:
            label += "_{}".format(category)
            fn += "_{}".format(category)

        fn += ".pdf"

        plt.figure()

        sns.scatterplot(x=distributions[0], y=distributions[1], label=label, legend=False, s=100)

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Degree")

        plt.tight_layout()

        image_path = os.path.join(images_folder, fn)
        plt.savefig(image_path)

        # plt.show()

        plt.clf()

    if distributions[3] is not None:
        label = "Users"
        fn = "degree_distributions_bottom"

        if population is not None:
            label += "_{}".format(population)
            fn += "_{}".format(population)

        fn += ".pdf"

        plt.figure()

        sns.scatterplot(x=distributions[2], y=distributions[3], label=label, legend=False, s=100)

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Degree")

        plt.tight_layout()

        image_path = os.path.join(images_folder, fn)

        plt.savefig(image_path)

        plt.clf()

    # plt.figure()


def compute_degree_distributions(data):

    bottom_grouped_df = data.groupby("User")  # users
    top_grouped_df = data.groupby("Item")  # items

    user_items = []
    item_users = []

    for _, group in bottom_grouped_df:
        user_items.append(len(group))

    for _, group in top_grouped_df:
        item_users.append(len(group))

    bottom_x, bottom_distribution = np.unique(user_items, return_counts=True)
    top_x, top_distribution = np.unique(item_users, return_counts=True)

    return top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users


def plot_category_percentages(percentages, saving_path, populations):

    percentages = np.array(percentages)

    colors = ["green", "blue", "red"]
    palette = {}

    for i, pop in enumerate(populations):
        palette[pop] = colors[i]

    fig, ax = plt.subplots()

    sns.histplot(
        x=percentages[:, 0].astype(np.float32),
        hue=percentages[:, 1].astype(np.int32),
        palette=palette,
        bins=20,
        kde=False,
        stat="probability",
    )

    ax.set_ylim([0.0, 0.5])

    plt.ylabel("Proportion")
    plt.tight_layout()

    plt.savefig(saving_path + "category_percentage_distribution.pdf")
    # plt.show()


def mu_sigma_to_alpha_beta(mu, sigma):
    """For Chaney's custom Beta' function, we convert
    a mean and variance to an alpha and beta parameter
    of a Beta function. See footnote 3 page 3 of Chaney
    et al. for details.
    """
    alpha = ((1 - mu) / (sigma**2) - (1 / mu)) * mu**2
    beta = alpha * (1 / mu - 1)
    return alpha, beta


def gen_dataset(params, dataset_path, relevance_sorting=0):
    # ALTERNATIVE SETTINGS
    article_mode = False

    # with open("config/{}.json".format(dataset)) as fp:
    #    params = json.load(fp)

    # parser = argparse.ArgumentParser()
    # t_args = argparse.Namespace()
    # t_args.__dict__.update(params)
    # params = parser.parse_args(args=None, namespace=t_args)

    # dataset_path = "dataset_test/{}/".format(params.dataset_name)

    σ = 1e-5
    num_attrs = 64  # K
    num_users = params.num_users
    num_items = params.num_items

    users_distribution = params.users_distribution
    users_distribution_xmax = params.users_distribution_xmax
    users_long_tail_alpha = params.users_long_tail_alpha
    users_long_tail_beta = params.users_long_tail_beta

    items_distribution = params.items_distribution
    items_distribution_xmax = params.items_distribution_xmax
    items_long_tail_alpha = params.items_long_tail_alpha
    items_long_tail_beta = params.items_long_tail_beta
    items_long_tail_coeff = params.items_long_tail_coeff

    MIN_LEN = params.min_length

    # ETA_users = [float(x) for x in params.ETA_users.split(",")]
    # ETA_items = [float(x) for x in params.ETA_items.split(",")]
    ETA_users = params.ETA_users
    ETA_items = params.ETA_items

    EPSILON = params.EPSILON

    # dataset_path += "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_epsilon_{}/".format(
    #    num_attrs,
    #    num_users,
    #    num_items,
    #    users_distribution,
    #    users_long_tail_alpha,
    #    users_long_tail_beta,
    #    items_distribution,
    #    items_long_tail_alpha,
    #    items_long_tail_beta,
    #    items_long_tail_coeff,
    #    EPSILON,
    # )
    # dataset_path += f"{dataset_ver:.2f}/"
    # if VERBOSE:
    #    print(dataset_path)
    # 0.6 % density

    sz_pwr = num_attrs

    num_populations = len(ETA_users)

    num_users_populations = [int(num_users * ETA_users[i]) for i in range(len(ETA_users))]

    users_populations_mapping = {}
    users = np.array(list(range(num_users)))
    old_n = 0
    for i, n in enumerate(num_users_populations):
        users_population_mapping_temp = dict.fromkeys(users[old_n : old_n + n], i)
        users_populations_mapping.update(users_population_mapping_temp)
        old_n += n

    if not exists(dataset_path):
        os.makedirs(dataset_path)

    # USERS

    ρ = []  # np.zeros((num_users, num_attrs))

    old_n = 0
    num_attr_per_pop = num_attrs // num_populations
    for i in range(num_populations):
        rng = np.random.default_rng(12121995)
        params = np.ones(num_attrs)
        if not article_mode:
            params = np.ones(num_attrs)
            if ETA_users[i] != 1.0:
                params[num_attr_per_pop * i : num_attr_per_pop * (i + 1)] = EPSILON
            μ_ρ = rng.dirichlet(params, size=num_users_populations[i]) * 10
        else:
            params = np.ones(num_attrs - num_attr_per_pop)
            μ_ρ = rng.dirichlet(params, size=num_users_populations[i]) * 10
            eps_section = np.ones((num_users_populations[i], num_attr_per_pop)) * EPSILON
            μ_ρ = np.concat((μ_ρ[:, 0 : i * num_attr_per_pop], eps_section, μ_ρ[:, i * num_attr_per_pop :]), axis=1)
        ρ_subpop = [rng.dirichlet(p) for p in μ_ρ]
        ρ.append(ρ_subpop)

    ρ = np.concatenate(ρ)

    num_categories = len(ETA_items)
    num_items_categories = [int(num_items * ETA_items[i]) for i in range(len(ETA_items))]

    items_categories_mapping = {}
    items = np.array(list(range(num_items)))
    old_n = 0
    for i, n in enumerate(num_items_categories):
        items_categories_mapping_temp = dict.fromkeys(items[old_n : old_n + n], i)
        items_categories_mapping.update(items_categories_mapping_temp)
        old_n += n

    α = []  # np.zeros((num_items, num_attrs))
    num_attr_per_cat = num_attrs // num_categories
    for i in range(num_categories):
        rng = np.random.default_rng(12121995)  # 12121995
        if not article_mode:
            params = np.ones(num_attrs) * 100

            if ETA_items[i] != 1.0:
                params[num_attr_per_cat * i : num_attr_per_cat * (i + 1)] = EPSILON

            μ_ρ = rng.dirichlet(params, size=num_items_categories[i]) * 0.1
        else:
            params = np.ones(num_attrs - num_attr_per_cat)
            μ_ρ = rng.dirichlet(params, size=num_items_categories[i]) * 10
            eps_section = np.ones((num_items_categories[i], num_attr_per_cat)) * EPSILON
            μ_ρ = np.concat((μ_ρ[:, 0 : i * num_attr_per_cat], eps_section, μ_ρ[:, i * num_attr_per_cat :]), axis=1)
        α_subcategory = [rng.dirichlet(p) for p in μ_ρ]

        α.append(np.array(α_subcategory))

    α = np.concatenate(α)

    rng = np.random.default_rng(12121995)
    # TRUE UTILS
    ρ_α = np.clip(ρ @ α.T, 1e-9, None)
    # sample total utility from a Beta distribution (equation 2)
    a, b = mu_sigma_to_alpha_beta(ρ_α, σ)

    V = rng.beta(a, b, size=(num_users, num_items))

    # ETA_users_str = str(ETA_users).replace("[", "").replace("]", "").replace(", ", "_")
    # ETA_items_str = str(ETA_items).replace("[", "").replace("]", "").replace(", ", "_")

    # saving_path = dataset_path + f"Populations_{ETA_users_str}_Categories_{ETA_items_str}/"
    saving_path = dataset_path

    if not exists(saving_path):
        os.makedirs(saving_path)

    # print(saving_path)

    # generating noisy factor eta (assumption 4)
    mu_eta = 0.98
    eta_alphas, eta_betas = mu_sigma_to_alpha_beta(mu_eta, σ)
    ω = rng.beta(eta_alphas, eta_betas, size=(num_users, num_items))

    # users noisy preferences p_users (equation 4)
    P = ω * V

    V_df = pd.DataFrame(V)

    def generate_history(user, p, min_len, g_pop, num_ones=0):
        items = []
        iteration = 0
        while len(items) < num_ones:
            if RNG_ALTERNATIVE or FASTER_RNG:
                rng_2 = np.random.default_rng(user + iteration)
                if FASTER_RNG:
                    check_val = rng_2.uniform()
            for item, p_i in enumerate(p):
                add = False
                if RNG_ALTERNATIVE:
                    add = rng_2.binomial(1, (p_i ** g_pop[item]))
                elif FASTER_RNG:
                    add = check_val <= p_i ** g_pop[item]
                else:
                    rng_2 = np.random.default_rng(user + iteration)
                    add = rng_2.binomial(1, (p_i ** g_pop[item]))
                if item not in items and add:
                    items.append(item)

            if user % 100 == 0 and VERBOSE:
                print("user", user, "iteration", iteration, "len items", len(items), "num_ones", num_ones)
            iteration += 1

            if iteration == 200:
                while len(items) < min_len:
                    for item, p_i in enumerate(p):
                        add = False
                        if RNG_ALTERNATIVE:
                            add = rng_2.binomial(1, (p_i ** g_pop[item]))
                        elif FASTER_RNG:
                            add = check_val <= p_i ** g_pop[item]
                        else:
                            rng_2 = np.random.default_rng(user + iteration)
                            add = rng_2.binomial(1, (p_i ** g_pop[item]))
                        if item not in items and add:
                            items.append(item)

                if VERBOSE:
                    print(
                        "maximum number of iterations user",
                        user,
                        "iteration",
                        iteration,
                        "len items",
                        len(items),
                        "num_ones",
                        num_ones,
                    )
                break

        if len(items) > num_ones:
            if not RNG_ALTERNATIVE:
                rng_2 = np.random.default_rng(user + iteration)
            indices = rng_2.choice(np.arange(len(items)), size=int(num_ones), replace=False)
            items = np.array(items)[indices]

        return items

    np.random.seed(12121995)
    g_pop = []
    dist = None
    if items_distribution == "power_law":
        dist = powerlaw.Power_Law(parameters=[items_long_tail_alpha])
        g_pop = dist.generate_random(num_items)
    elif items_distribution == "power_law_with_cutoff":
        dist = powerlaw.Truncated_Power_Law(parameters=[items_long_tail_alpha, items_long_tail_beta])
        g_pop = dist.generate_random(num_items)
    elif items_distribution == "stretched_exponential":
        dist = powerlaw.Stretched_Exponential(parameters=[items_long_tail_alpha, items_long_tail_beta])
        g_pop = dist.generate_random(num_items)
    elif items_distribution == "log_normal":
        dist = powerlaw.Lognormal(parameters=[items_long_tail_alpha, items_long_tail_beta])
        g_pop = dist.generate_random(num_items)

    g_pop = dist._pdf_base_function(g_pop) * dist._pdf_continuous_normalizer

    if items_long_tail_coeff == 0.0:
        g_pop = np.ones_like(g_pop)
    else:
        g_pop = items_long_tail_coeff * (1 - g_pop)

    # OWN CODE
    if relevance_sorting != 0:
        population_mask = np.array([users_populations_mapping[i] for i in range(num_users)])
        relevance_means = np.zeros((num_populations, num_items))
        for i in range(num_populations):
            sub_relevances = V[population_mask == i]
            relevance_means[i, :] = sub_relevances.mean(0)
        relevance_var = np.var(relevance_means, 0)
        sorted_positions = np.argsort(relevance_sorting * relevance_var)
        pop_ordering = np.argsort(-g_pop)
        sorted_g_pop = np.ones_like(g_pop)
        for i, destination_position in enumerate(sorted_positions):
            sorted_g_pop[destination_position] = g_pop[pop_ordering[i]]
        g_pop = sorted_g_pop
    num_ones = np.zeros(num_users)

    np.random.seed(12121995)
    if users_distribution == "power_law":
        dist = powerlaw.Power_Law(parameters=[users_long_tail_alpha])
        num_ones = dist.generate_random(num_users).astype(np.int32)
    elif users_distribution == "power_law_with_cutoff":
        dist = powerlaw.Truncated_Power_Law(parameters=[users_long_tail_alpha, users_long_tail_beta])
        num_ones = dist.generate_random(num_users).astype(np.int32)
    elif users_distribution == "stretched_exponential":
        dist = powerlaw.Stretched_Exponential(parameters=[users_long_tail_alpha, users_long_tail_beta])
        num_ones = dist.generate_random(num_users).astype(np.int32)
    elif users_distribution == "log_normal":
        dist = powerlaw.Lognormal(parameters=[users_long_tail_alpha, users_long_tail_beta])
        num_ones = dist.generate_random(num_users).astype(np.int32)

    if users_distribution_xmax != -1:
        for i in range(len(num_ones)):
            while num_ones[i] > users_distribution_xmax:
                num_ones[i] = dist.generate_random(1).astype(np.int32)[0]

    num_ones += MIN_LEN

    num_ones_final = []
    for n_u in num_ones:
        while n_u >= num_items:
            n_u = dist.generate_random(1).astype(np.int32)[0]
            n_u += MIN_LEN
        num_ones_final.append(n_u)

    num_ones = num_ones_final

    histories = Parallel(n_jobs=10, prefer="processes")(
        delayed(generate_history)(user, p, MIN_LEN, g_pop, num_ones[user]) for user, p in enumerate(P)
    )
    # histories = []
    # for user, p in enumerate(P):
    #    histories.append(generate_history(user, p, MIN_LEN, g_pop, num_ones[user]))

    # OWN CODE!!!!
    dump_matrices(dataset_path, histories, num_users, num_items, num_users_populations)

    interactions = []
    percentages = []
    for user, history in enumerate(histories):

        count = 0

        for item in history:

            interactions.append((user, item, users_populations_mapping[user], items_categories_mapping[item]))
            if items_categories_mapping[item] == 0:
                count += 1

        perc = count / len(history) if len(history) > 0 else 0.0
        percentages.append([perc, users_populations_mapping[user]])

    df = pd.DataFrame(columns=["User", "Item", "Population", "Category"], data=interactions)

    user_items, item_users = [], []
    if VERBOSE:
        print(f"Num users {len(df['User'].unique())}")
        print(f"Num items {len(df['Item'].unique())}")
        print(f"Num interazioni {df.shape[0]}")
    if (len(df["User"].unique()) * len(df["Item"].unique())) > 0:
        if VERBOSE:
            print(f"Densità {df.shape[0] / (len(df['User'].unique()) * len(df['Item'].unique()))}")

            print(saving_path)
        show_interactions_plot(df, saving_path)
        # print(df)

        top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users = compute_degree_distributions(
            df
        )
        plot_degree_distributions([top_x, top_distribution, bottom_x, bottom_distribution], saving_path)

        if len(ETA_users) > 1:
            for pop in range(len(ETA_users)):
                temp_df = df[df["Population"] == pop]
                top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users = (
                    compute_degree_distributions(temp_df)
                )
                plot_degree_distributions([None, None, bottom_x, bottom_distribution], saving_path, population=pop)

        if len(ETA_items) > 1:
            for cat in range(len(ETA_items)):
                temp_df = df[df["Category"] == cat]
                top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users = (
                    compute_degree_distributions(temp_df)
                )
                plot_degree_distributions([top_x, top_distribution, None, None], saving_path, category=cat)

                if cat == 0 and len(ETA_items) > 1:
                    plot_category_percentages(percentages, saving_path, np.arange(len(ETA_users)))

    df.to_csv(saving_path + "Histories.tsv", index=False, sep="\t")
    if VERBOSE:
        print("Finished")


def dump_matrices(dataset_path, histories, num_users, num_items, num_users_populations, min_occ=5, min_items=5):
    start = 0
    s = np.zeros((num_users, 2))
    tr_inds = np.array([], dtype=int)
    va_inds = np.array([], dtype=int)
    te_inds = np.array([], dtype=int)
    for i, n in enumerate(num_users_populations):
        end = start + n
        s[start:end, :] = i

        # Strat
        inds = np.arange(start, end)
        np.random.shuffle(inds)
        tr_inds = np.concat((tr_inds, inds[0 : int(0.7 * n)]))
        va_inds = np.concat((va_inds, inds[int(0.7 * n) : int(0.85 * n)]))
        te_inds = np.concat((te_inds, inds[int(0.85 * n) :]))

        # Update current ind
        start = end

    tr_s = s[tr_inds, :]
    x = np.zeros((num_users, num_items))
    for i, history in enumerate(histories):
        x[i, history] = 1
    tr_x = x[tr_inds]
    tr_s = s[tr_inds]

    # Mask out items with few occurences and users with few items
    item_mask = tr_x.sum(0) >= min_occ
    tr_x = tr_x[:, item_mask]
    tr_user_mask = tr_x.sum(1) >= min_items
    tr_x = tr_x[tr_user_mask]
    tr_s = tr_s[tr_user_mask]

    # Split up val and test
    def split_data(matrix, labels, item_mask, min_items):
        x = np.zeros(matrix.shape)
        y = np.zeros(matrix.shape)
        for i, ints in enumerate(matrix):
            items = ints.nonzero()[0]
            np.random.shuffle(items)
            x[i, items[: int(items.shape[0] * 0.8)]] = 1
            y[i, items[int(items.shape[0] * 0.8) :]] = 1

        # Mask out items and users
        x = x[:, item_mask]
        y = y[:, item_mask]
        user_mask = x.sum(1) >= min_items
        x = x[user_mask]
        y = y[user_mask]
        labels = labels[user_mask]

        return x, y, labels

    va_x, va_y, va_s = split_data(x[va_inds], s[va_inds, :], item_mask, min_items)
    te_x, te_y, te_s = split_data(x[te_inds], s[te_inds, :], item_mask, min_items)

    np.savez(
        f"{dataset_path}synth.npz",
        tr_x=tr_x,
        tr_s=tr_s,
        va_x=va_x,
        va_y=va_y,
        va_s=va_s,
        te_x=te_x,
        te_y=te_y,
        te_s=te_s,
    )


# if __name__ == "__main__":
#    dataset = "log_norm_test"
#    main(dataset)
