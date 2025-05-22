import numpy as np
import json
from dataclasses import dataclass

from genrec_orig import gen_dataset
from seq_classify import eval_classify
from item2vec import fit_item2vec

from optin.data_processing import load_and_uncompress


def main():
    params = DatasetParams()
    n_tests = 5

    auc_info = True  # Test VAEAfrl* with different hyperparams
    auc_perturb = True  # Test VAERel/VAE2adv with different hyperparams
    eps_test = False  # Test models on synthetic datasets with different epsilon
    n_users_test = False  # Test models on synthetic datasets with different number of users
    skewdness_test = False  # Test models on synthetic datasets with different ratios of minority users
    synth = True  # Set to False for Movielens
    s_ind = 0  # Set to 1 for Movielens Age (0 for synthetic or Movielens Gender)
    double_adv = False  # Set to True for VAE2adv instead of VAERel
    device = "cuda"

    # Embedding params
    min_count = 5
    emb_tr_batch = 64
    embedding_dim = 8
    dim2 = 128
    verbose = False

    dataset_filename = "synth.npz"

    # Rec AUC vs Rep AUC
    if auc_info:
        # Info align
        params = DatasetParams()
        params.EPSILON = 0.62
        dataset = "Info_Synth_062"
        lambdas = [0.4 + 0.2 * i for i in range(79)]
        n_advs = [10] * len(lambdas)
        for i, val in enumerate(lambdas):
            if val >= 5:
                n_advs[i] = 10 + int(val - 4)
        # lambdas = [6.5, 8.0, 10.0, 15.0, 20.0, 40.0]
        # n_advs = [15, 15, 20, 20, 22, 25]
        # lambdas = [20.0, 40.0]
        # n_advs = [20, 30]
        test_results = []
        test_mean_results = []
        if synth:
            dataset_path = f"grid_tests/{dataset}/"
            gen_dataset(params, dataset_path)
            tr_x = np.load(dataset_path + dataset_filename)["tr_x"]
        else:
            dataset_path = "movielens/"
            outputs = load_and_uncompress(dataset_path, user_split=True)
            tr_x = outputs[0]

        pretrained_embeddings = []
        for i in range(n_tests):
            embeddings = fit_item2vec(
                tr_x, min_count, emb_tr_batch, embedding_dim, dim2, device, dataset_path, verbose
            )
            pretrained_embeddings.append(embeddings)
        for j, lambd in enumerate(lambdas):
            print(f"\nLambda {lambd}")
            results = []
            for i in range(n_tests):
                embeddings = pretrained_embeddings[i]
                result = eval_classify(
                    dataset_path,
                    dataset_filename,
                    budget=-1,
                    info_lambd=lambd,
                    info_adv=n_advs[j],
                    embeddings=embeddings,
                    synth=synth,
                    s_ind=s_ind,
                )
                results.append(result)
            all_results = {}
            for model_key in results[0].keys():
                model_map = {}
                for attr_key in results[0][model_key].keys():
                    model_map[attr_key] = [m[model_key][attr_key] for m in results]
                all_results[model_key] = model_map

            mean_results = {}
            for model_key in all_results.keys():
                model_map = {}
                for attr_key in all_results[model_key].keys():
                    model_map[attr_key] = np.mean(all_results[model_key][attr_key])
                mean_results[model_key] = model_map
            test_results.append(all_results)
            test_mean_results.append(mean_results)
            print_results(all_results)
            with open(dataset_path + f"info{lambd:.1f}.json", "w") as f:
                json.dump(all_results, f)

    if auc_perturb:
        # Perturb
        params = DatasetParams()
        dataset = "Perturb_Synth"
        # budget = [1, 2, 5, 10, 20, 35, 50, 65, 80, 100, 150]
        budget = list(range(1, 26))
        # budget = list(range(37, 101, 2))
        test_results = []
        test_mean_results = []
        if synth:
            dataset_path = f"grid_tests/{dataset}/"
            gen_dataset(params, dataset_path)
            tr_x = np.load(dataset_path + dataset_filename)["tr_x"]
        else:
            dataset_path = "movielens/"
            outputs = load_and_uncompress(dataset_path, user_split=True)
            tr_x = outputs[0]

        pretrained_embeddings = []
        for i in range(n_tests):
            embeddings = fit_item2vec(
                tr_x, min_count, emb_tr_batch, embedding_dim, dim2, device, dataset_path, verbose
            )
            pretrained_embeddings.append(embeddings)
        for bud in budget:
            print(f"\nBudget {bud}")
            results = []
            for i in range(n_tests):
                embeddings = pretrained_embeddings[i]
                result = eval_classify(
                    dataset_path,
                    dataset_filename,
                    budget=bud,
                    info_lambd=-1,
                    embeddings=embeddings,
                    synth=synth,
                    s_ind=s_ind,
                    double_adv=double_adv,
                )
                results.append(result)
            all_results = {}
            for model_key in results[0].keys():
                model_map = {}
                for attr_key in results[0][model_key].keys():
                    model_map[attr_key] = [m[model_key][attr_key] for m in results]
                all_results[model_key] = model_map

            mean_results = {}
            for model_key in all_results.keys():
                model_map = {}
                for attr_key in all_results[model_key].keys():
                    model_map[attr_key] = np.mean(all_results[model_key][attr_key])
                mean_results[model_key] = model_map
            test_results.append(all_results)
            test_mean_results.append(mean_results)
            print_results(all_results)
            with open(dataset_path + f"perturb{bud}.json", "w") as f:
                json.dump(all_results, f)

    # Epsilon
    if eps_test:
        params = DatasetParams()
        dataset = "eps_hires_both"
        # n_us = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
        n_us = 0.02 * np.arange(1, 51)
        test_results = []
        test_mean_results = []
        for n_u in n_us:
            print(f"\nEps {n_u}")
            dataset_path = f"grid_tests/{dataset}/{n_u:.2f}/"
            params.EPSILON = n_u

            results = []
            for i in range(n_tests):
                gen_dataset(params, dataset_path)
                tr_x = np.load(dataset_path + dataset_filename)["tr_x"]
                embeddings = fit_item2vec(
                    tr_x, min_count, emb_tr_batch, embedding_dim, dim2, device, dataset_path, verbose
                )
                result = eval_classify(
                    dataset_path,
                    dataset_filename,
                    embeddings=embeddings,
                    budget=20,
                    info_lambd=-1,
                    double_adv=double_adv,
                )
                results.append(result)
            all_results = {}
            for model_key in results[0].keys():
                model_map = {}
                for attr_key in results[0][model_key].keys():
                    model_map[attr_key] = [m[model_key][attr_key] for m in results]
                all_results[model_key] = model_map

            mean_results = {}
            for model_key in all_results.keys():
                model_map = {}
                for attr_key in all_results[model_key].keys():
                    model_map[attr_key] = np.mean(all_results[model_key][attr_key])
                mean_results[model_key] = model_map
            test_results.append(all_results)
            test_mean_results.append(mean_results)
            print_results(all_results)
            with open(dataset_path + "results.json", "w") as f:
                json.dump(all_results, f)

    # N_users
    if n_users_test:
        params = DatasetParams()
        dataset = "n_u4000_new"
        n_us = [3000, 4000, 6000, 8000, 12000, 16000]
        test_results = []
        test_mean_results = []
        for n_u in n_us:
            print(f"\nNum users {n_u}")
            dataset_path = f"grid_tests/{dataset}/{n_u:.2f}/"
            params.num_users = n_u
            # params.users_distribution_xmax = n_u
            params.EPSILON = 0.5
            params.ETA_users = [0.5, 0.5]

            results = []
            for i in range(n_tests):
                gen_dataset(params, dataset_path)
                tr_x = np.load(dataset_path + dataset_filename)["tr_x"]
                embeddings = fit_item2vec(
                    tr_x, min_count, emb_tr_batch, embedding_dim, dim2, device, dataset_path, verbose
                )
                result = eval_classify(
                    dataset_path, dataset_filename, embeddings=embeddings, info_lambd=-1, double_adv=double_adv
                )
                results.append(result)
            all_results = {}
            for model_key in results[0].keys():
                model_map = {}
                for attr_key in results[0][model_key].keys():
                    model_map[attr_key] = [m[model_key][attr_key] for m in results]
                all_results[model_key] = model_map

            mean_results = {}
            for model_key in all_results.keys():
                model_map = {}
                for attr_key in all_results[model_key].keys():
                    model_map[attr_key] = np.mean(all_results[model_key][attr_key])
                mean_results[model_key] = model_map
            test_results.append(all_results)
            test_mean_results.append(mean_results)
            print_results(all_results)
            with open(dataset_path + "results.json", "w") as f:
                json.dump(all_results, f)

    # Skewdness
    if skewdness_test:
        params = DatasetParams()
        dataset = "skewdness"
        skewdness_ratio = 0.01 * np.arange(1, 51)
        test_results = []
        test_mean_results = []
        for s_rat in skewdness_ratio:
            print(f"\nMinority ratio {s_rat:.3f}")
            dataset_path = f"grid_tests/{dataset}/{s_rat:.3f}/"
            params.ETA_users = [s_rat, 1.0 - s_rat]

            results = []
            for i in range(n_tests):
                gen_dataset(params, dataset_path)
                tr_x = np.load(dataset_path + dataset_filename)["tr_x"]
                embeddings = fit_item2vec(
                    tr_x, min_count, emb_tr_batch, embedding_dim, dim2, device, dataset_path, verbose
                )
                result = eval_classify(
                    dataset_path, dataset_filename, embeddings=embeddings, info_lambd=-1, double_adv=double_adv
                )
                results.append(result)
            all_results = {}
            for model_key in results[0].keys():
                model_map = {}
                for attr_key in results[0][model_key].keys():
                    model_map[attr_key] = [m[model_key][attr_key] for m in results]
                all_results[model_key] = model_map

            mean_results = {}
            for model_key in all_results.keys():
                model_map = {}
                for attr_key in all_results[model_key].keys():
                    model_map[attr_key] = np.mean(all_results[model_key][attr_key])
                mean_results[model_key] = model_map
            test_results.append(all_results)
            test_mean_results.append(mean_results)
            print_results(all_results)
            with open(dataset_path + "results.json", "w") as f:
                json.dump(all_results, f)
    print("Finished")


def print_results(all_results):
    for model_name, model_map in all_results.items():
        eps = 1.0e-7
        model_str = f"{model_name}: "
        for metric, metric_vals in model_map.items():
            model_str += f"{metric}: {np.mean(metric_vals):.4f} "
            metric_std = np.std(metric_vals)
            if metric_std > eps:
                model_str += f"+- {metric_std:.4f} "
        print(model_str)


class DatasetParams:
    dataset_name: str = ""
    users_distribution: str = "log_normal"
    users_long_tail_alpha: float = 4.09
    users_long_tail_beta: float = 0.98
    users_distribution_xmax: int = -1
    items_distribution: str = "log_normal"
    items_long_tail_alpha: float = 3.5
    items_long_tail_beta: float = 1.9
    items_distribution_xmax: int = -1
    items_long_tail_coeff: float = 2.0
    num_users: int = 4000
    num_items: int = 4000
    min_length: int = 0
    ETA_users: list[float] = [0.3, 0.7]
    ETA_items: list[float] = [0.5, 0.5]
    EPSILON: float = 0.5


if __name__ == "__main__":
    main()
