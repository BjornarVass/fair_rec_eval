import numpy as np
from dataclasses import dataclass

from genrec_orig import gen_dataset
from seq_classify import eval_classify


def main():
    params = DatasetParams()
    n_tests = 5

    eps_test = True
    n_users_test = False

    # Epsilon
    if eps_test:
        dataset = "eps2"
        n_us = [0.05, 0.2, 0.5, 1.0]
        test_results = []
        test_mean_results = []
        for n_u in n_us:
            print(f"\nEps {n_u}")
            dataset_path = f"grid_tests/{dataset}/{n_u:.2f}/"
            params.EPSILON = n_u
            gen_dataset(params, dataset_path, relevance_sorting=1)

            results = []
            for i in range(n_tests):
                result = eval_classify(dataset_path, "synth.npz")
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

    # N_users
    if n_users_test:
        dataset = "n_u"
        n_us = [250, 500, 1000, 2000]
        test_results = []
        test_mean_results = []
        for n_u in n_us:
            print(f"\nNum users {n_u}")
            dataset_path = f"grid_tests/{dataset}/{n_u:.2f}/"
            params.num_users = n_u
            params.users_distribution_xmax = n_u
            params.EPSILON = 1.0
            gen_dataset(params, dataset_path)

            results = []
            for i in range(n_tests):
                result = eval_classify(dataset_path, "synth.npz")
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
    print("lawl")


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
    users_distribution_xmax: int = 500
    items_distribution: str = "log_normal"
    items_long_tail_alpha: float = 3.5
    items_long_tail_beta: float = 1.9
    items_distribution_xmax: int = -1
    items_long_tail_coeff: float = 2.0
    num_users: int = 500
    num_items: int = 500
    min_length: int = 0
    ETA_users: list[float] = [0.3, 0.7]
    ETA_items: list[float] = [0.5, 0.5]
    EPSILON: float = 0.5


if __name__ == "__main__":
    main()
