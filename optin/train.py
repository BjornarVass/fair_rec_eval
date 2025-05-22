import numpy as np
import wandb
import argparse

import torch
import torch.utils.data
from collections import Counter
from torch import optim
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from scipy.sparse import csr_matrix, vstack

from optin.data_processing import (
    init_dataloader,
    preprocess_movielens,
    preprocess_lastfm,
    load_and_uncompress,
)
from optin.model import (
    FairVAERec,
    ModelSettings,
    RunSettings,
    VAEafrlSettings,
    PerturbSettings,
    EvalStorage,
    EvalResults,
)
from optin.perturb import perturb_model, perturb_users
from optin.vae_afrl import train_info_align, info_align_encode
from optin.utils import (
    ndcg_at_k,
    generate_network_dims,
    dump_torch_ratio,
    update_early_stopping,
)


def parse_arguments(args=None):
    bool_t = "store_true"
    parser = argparse.ArgumentParser(description="Process arguments")
    # RUN
    parser.add_argument("--run_group", type=str, default="RUN", help="Specify wandb run group name")
    parser.add_argument(
        "--dataset",
        type=str,
        default="movielens",
        help="Options: 'movielens' and 'lastfm'",
    )
    parser.add_argument("--verbose", default=False, action=bool_t, help="Enable logging")
    parser.add_argument("--debug", default=False, action=bool_t, help="Enable various debug printing")
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=-1,
        help="Early stopping: number of epochs without improvement before stopping the training. Disabled when set to -1",
    )
    parser.add_argument("--n_adv_train", type=int, default=1, help="# adversarial updates per minibatch")
    parser.add_argument("--n_epochs", type=int, default=1, help="# training epochs")
    parser.add_argument("--n_adv_pre", type=int, default=150, help="# adversarial pre-train epochs")
    parser.add_argument("--tr_batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--nbatch_per_update", type=int, default=1, help="# batches per model updates")
    parser.add_argument("--te_batch_size", type=int, default=1, help="Testing batch size")
    parser.add_argument(
        "--train_eval",
        default=False,
        action=bool_t,
        help="Enable evaluation during training",
    )
    parser.add_argument(
        "--rep_eval_interval",
        type=int,
        default=40,
        help="# epochs between evaluating Representation Neutrality",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Name of the device used for computation",
    )
    parser.add_argument("--save_model", default=False, action=bool_t, help="Save model")
    parser.add_argument("--load_model", default=False, action=bool_t, help="Load model, skip training")

    # VAEAFRL*
    parser.add_argument("--info_align", default=False, action=bool_t, help="Enable VAEafrl*")
    parser.add_argument(
        "--afrl_beta",
        type=float,
        default=0.1,
        help="VAEafrl* beta weight. Adversarial loss weight when using attribute encoders",
    )
    parser.add_argument("--lambd", type=float, default=1.0, help="VAEafrl* loss parameter lambda")
    parser.add_argument("--afrl_weight_decay", type=float, default=1e-8, help="VAEafrl* weight decay")
    parser.add_argument("--afrl_lr", type=float, default=5e-5, help="VAEafrl* learning rate")
    parser.add_argument("--n_encode_epochs", type=int, default=1000, help="# VAEafrl encoder+adv training epochs")
    parser.add_argument(
        "--n_combine_epochs", type=int, default=500, help="# VAEafrl combination network training epochs"
    )

    # PERTURB
    parser.add_argument("--perturb", default=False, action=bool_t, help="Enable perturb")
    parser.add_argument(
        "--grad_only", default=False, action=bool_t, help="Only consider fairness when augmenting user data"
    )
    parser.add_argument("--double_adv", default=False, action=bool_t, help="Enable double adversarial")
    parser.add_argument("--max_budget", type=int, default=50, help="Max budget, perturb models")
    parser.add_argument(
        "--adv_prob_threshold",
        type=float,
        default=0.025,
        help="Threshold used for terminating pertubations when close to ideal adv probs",
    )

    # VAESM
    parser.add_argument("--filter", default=False, action=bool_t, help="Enable VAEsm")
    parser.add_argument("--gamma", type=float, default=1.0, help="VAEsm loss parameter gamma")

    # PREPROCESSING AND DATA
    # Note: Integrated preprocessing has limited support and assumes standard filenames.
    parser.add_argument(
        "--preprocess",
        default=False,
        action=bool_t,
        help="Enable integrated data preprocessing",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="PREPROCESSING: Path to data directory",
    )
    parser.add_argument(
        "--item_split",
        default=False,
        action=bool_t,
        help="Split data on items, not users",
    )
    parser.add_argument(
        "--imbalanced_sensitive",
        default=False,
        action=bool_t,
        help="Disable demographic data stratification",
    )
    parser.add_argument(
        "--valtest_frac",
        type=float,
        default=0.2,
        help="Fraction of users/items in valtest. Valtest is split 50/50",
    )
    # Note: Assumes processed file names
    parser.add_argument("--processed_dir", type=str, default="pro_ml/0/", help="Path to processsed data")
    parser.add_argument(
        "--csr",
        default=False,
        action=bool_t,
        help="Load data as csr to reduce memory impact",
    )
    parser.add_argument(
        "--load_synth",
        default=False,
        action=bool_t,
        help="Enable alternative loading of synth data",
    )

    # MODEL STRUCTURE
    parser.add_argument(
        "--hidden_activation",
        type=str,
        default="selu",
        help="Options: 'selu', 'tanh' and 'relu'",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=500,
        help="Hidden dimensionality of encoder and decoder",
    )
    parser.add_argument(
        "--hidden_dim_adv",
        type=int,
        default=16,
        help="Hidden dimensionality of adversarial models",
    )
    parser.add_argument("--n_hidden_vae", type=int, default=1, help="# hidden layers in VAE")
    parser.add_argument(
        "--n_hidden_adv",
        type=int,
        default=1,
        help="# hidden layers in adversarial models",
    )
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of latent state")
    parser.add_argument("--adv", default=False, action=bool_t, help="Enable adversarial")
    parser.add_argument("--deep_adv", default=False, action=bool_t, help="Enable deep adversarial")
    parser.add_argument("--dropout", type=float, default=0.2, help="Model dropout rate")
    parser.add_argument("--dropout_adv", type=float, default=0.0, help="Adversarial dropout rate")
    parser.add_argument(
        "--noise_rate",
        type=float,
        default=0.5,
        help="Fraction of items dropped during training",
    )

    # MODEL OPTIMIZATION
    parser.add_argument("--lr", type=float, default=0.001, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument("--beta", type=float, default=1.0, help="Loss parameter beta")

    # EVALUATION
    parser.add_argument("--k", type=int, default=10, help="# of ranks considered in NDCG metric")
    if args is not None:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    return run(args)


def run(args):
    # Pre-process datasets
    user_split = not args.item_split
    if args.preprocess:
        preprocess_datasets(args, user_split)
        exit()

    # Load datasets
    (
        train_data,
        val_tr,
        val_te,
        test_tr,
        test_te,
        sensitive_labels,
        train_s,
        val_s,
        test_s,
        title_map,
    ) = load_and_uncompress(args.processed_dir, user_split=user_split, csr=args.csr, synth=args.load_synth)

    # Define network structures based processed data and hyperparameters
    n_sensitive = len(sensitive_labels)
    n_items = train_data.shape[1]
    sensitive_counts = train_s.sum(0)
    demographics = sensitive_counts / train_s.shape[0]
    z_enc_dims = generate_network_dims(n_items, args.hidden_dim, 2 * args.latent_dim, n_hidden=args.n_hidden_vae)
    z_dec_dims = generate_network_dims(args.latent_dim, args.hidden_dim, n_items, n_hidden=args.n_hidden_vae)

    adv_dims = generate_network_dims(args.latent_dim, args.hidden_dim_adv, 1, n_hidden=args.n_hidden_adv)

    # Init data loaders
    if user_split:
        train_loader = init_dataloader(
            train_data, None, train_s, args.device, args.tr_batch_size, args.csr, shuffle=True
        )
        val_loader = init_dataloader(val_tr, val_te, val_s, args.device, args.te_batch_size, args.csr)
        test_loader = init_dataloader(test_tr, test_te, test_s, args.device, args.te_batch_size, args.csr)
    else:
        train_loader = init_dataloader(
            train_data, val_te, train_s, args.device, args.tr_batch_size, args.csr, shuffle=True
        )
        val_loader = train_loader
        test_loader = init_dataloader(train_data, test_te, train_s, args.device, args.te_batch_size, args.csr)

    # Settings
    model_settings = ModelSettings(
        z_enc_dims=z_enc_dims,
        z_dec_dims=z_dec_dims,
        adv_dims=adv_dims,
        noise_rate=0.5,
        dropout=args.dropout,
        dropout_adv=args.dropout_adv,
        beta=args.beta,
        gamma=args.gamma,
        adv=args.adv or args.filter or args.perturb,
        deep_adv=args.deep_adv,
        adv_thresholds=None,
        filter=args.filter,
        latent_dim=args.latent_dim,
        n_sensitive=n_sensitive,
        sensitive_labels=sensitive_labels,
        hidden_activation=args.hidden_activation,
        device=args.device,
        verbose=args.verbose,
    )

    # Start logging
    if args.verbose:
        hyperparams = vars(model_settings)
        hyperparams["dataset"] = args.dataset
        wandb.init(group=args.run_group, config=hyperparams)

    # Collect run settings for model training
    run_settings = RunSettings(
        model=None,
        opt_model=None,
        opt_adv=None,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        n_adv_pre=args.n_adv_pre,
        nbatch_per_update=args.nbatch_per_update,
        n_adv_train=args.n_adv_train,
        k=args.k,
        train_eval=args.train_eval,
        rep_eval_interval=args.rep_eval_interval,
        csr=args.csr,
        title_map=title_map,
        processed_dir=args.processed_dir,
        demographics=demographics,
        debug=args.debug,
        early_stopping=args.early_stopping,
        info_align=args.info_align,
        perturb=args.perturb,
    )

    # VAEafrl* settings
    if run_settings.info_align:
        run_settings.afrl = VAEafrlSettings(
            beta=args.afrl_beta,
            lambd=args.lambd,
            weight_decay=args.afrl_weight_decay,
            lr=args.afrl_lr,
            n_encode_epochs=args.n_encode_epochs,
            n_combine_epochs=args.n_combine_epochs,
        )

    # Perturb settings
    if run_settings.perturb:
        run_settings.perturb_settings = PerturbSettings(
            grad_only=args.grad_only,
            double_adv=args.double_adv,
            max_budget=args.max_budget,
            adv_prob_threshold=args.adv_prob_threshold,
        )

    if not args.load_model:
        # Init model and optimizers
        model, opt_model, opt_adv = init_model_and_optimizers(
            model_settings,
            args.weight_decay,
            adv=args.adv,
            afrl=run_settings.info_align,
            double_adv=args.double_adv,
        )
        run_settings.model = model
        run_settings.opt_model = opt_model
        run_settings.opt_adv = opt_adv

        # Train model
        # Turn off logging during training if train_eval is set to false
        verbose_setting = model.c.verbose
        if not args.train_eval:
            model.c.verbose = False
        train_model(run_settings)
        # Revert verbose setting
        model.c.verbose = verbose_setting

        # Train adversarial model
        model.eval()
        if not model.c.filter and model.c.adv:
            adv_log = {}
            model.c.adv_thresholds = train_adv(run_settings, adv_log, add_noise=run_settings.perturb)
            if args.double_adv:
                run_settings.opt_adv = optim.Adam(
                    model.adversarials2.parameters(), lr=0.001, weight_decay=args.weight_decay
                )
                model.c.adv_thresholds2 = train_adv(
                    run_settings, adv_log, add_noise=run_settings.perturb, perturb=run_settings.perturb
                )

        if args.verbose and not args.filter:
            test_log = {}
            run_settings.model.eval()
            run_settings.val_loader = test_loader
            test_log = {}
            evaluate_model(
                run_settings,
                test_log,
                eval_representations=True,
            )
            wandb.log(test_log)
            wandb.finish()

        if args.save_model:
            torch.save(model, args.processed_dir + f"/model.pt")
            exit()
    else:
        model = torch.load(args.processed_dir + f"/model.pt")
        run_settings.model = model
        if args.verbose:
            model.c.verbose = True

    # Train info align extension
    if run_settings.info_align:
        train_info_align(run_settings)

    # Evaluate on test set
    # Set model to eval mode and replace validation loader with the test set
    test_log = {}
    run_settings.model.eval()
    run_settings.val_loader = test_loader
    # inference(run_settings, test_log, perturb=True)

    if run_settings.info_align:
        evaluate_masks(run_settings, test_log)
    elif model.c.filter:
        evaluate_masks(run_settings, test_log, info_align=False)
    elif run_settings.perturb:
        evaluate_model(
            run_settings,
            test_log,
            eval_representations=True,
            perturb=True,
        )
    train_loader_no_shuffle = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False
    )
    base_probs, fair_probs, base_auc, fair_auc = dump_probs(
        run_settings,
        [train_loader_no_shuffle, val_loader, test_loader],
        2,
        perturb=run_settings.perturb,
        info_align=run_settings.info_align,
        filter=model.c.filter,
    )

    # Log test results and close log
    if model.c.verbose:
        wandb.log(test_log)
        wandb.finish()

    return base_probs, fair_probs, base_auc, fair_auc


def preprocess_datasets(args, user_split):
    balance_sensitive = not args.imbalanced_sensitive
    if args.dataset == "movielens":
        ratings_filename = "ratings.dat"
        user_info_filename = "users.dat"
        movie_info_filename = "movies.dat"
        preprocess_movielens(
            args.data_dir,
            args.processed_dir,
            ratings_filename,
            user_info_filename,
            args.valtest_frac,
            balance_sensitive=balance_sensitive,
            user_split=user_split,
            movie_filename=movie_info_filename,
        )

    elif args.dataset == "lastfm":
        ratings_filename = "lastfm.tsv"
        user_info_filename = "users.tsv"
        album_filename = "albums.tsv"
        preprocess_lastfm(
            args.data_dir,
            args.processed_dir,
            ratings_filename,
            user_info_filename,
            album_filename,
            args.valtest_frac,
            balance_sensitive=balance_sensitive,
            user_split=user_split,
        )
    else:
        raise ValueError(f"Unsupported dataset: '{args.dataset}'")


def init_model_and_optimizers(model_settings, weight_decay, adv=False, afrl=False, double_adv=False):
    if afrl:
        model = FairVAERec(model_settings, info_align=True)
    elif double_adv:
        model = FairVAERec(model_settings, double_adv=True)
    else:
        model = FairVAERec(model_settings)
    # print(model)
    model_params = model.get_params()
    opt_model = optim.Adam(model_params, lr=0.001, weight_decay=weight_decay)

    if not adv:
        return model, opt_model, None

    # Add adversarial optimizer
    adv_params = model.get_params(adversarial=True)

    opt_adv = optim.Adam(adv_params, lr=0.001, weight_decay=weight_decay)

    return model, opt_model, opt_adv


def train_model(settings):
    model = settings.model
    model.to(model.c.device)
    virtual_counter = 1
    early_stopping = False
    if settings.early_stopping > 0:
        n_es = settings.early_stopping
        early_stopping = True
        best_loss = np.inf
        best_model = None
        es_counter = 0
        es_eps = 1e-5
    upper_epoch = settings.n_epochs if not early_stopping else 1000
    epoch = 0
    while epoch < upper_epoch:
        # Set periodic flags
        eval_representations = epoch != 0 and epoch % settings.rep_eval_interval == 0

        # Log dict for wandb
        epoch_logs = {}

        # Batch storage
        batch_storage = EvalStorage()

        # MAIN TRAINNG LOOP
        model.train()
        for i, (x, _, s, _) in enumerate(settings.train_loader):
            # Move batch to GPU if Csr is used
            if settings.csr:
                x = x.to(model.c.device)
                s = s.to(device=model.c.device, dtype=torch.float32)

            # Train VAE
            if virtual_counter < settings.nbatch_per_update:
                run_step = False
                virtual_counter += 1
            else:
                run_step = True
            mask = np.random.randint(2, size=model.c.n_sensitive) if model.c.filter else None
            z, _, _, _ = call_fw_bw(settings, x, s, epoch_logs, step=run_step, mask=mask)

            # Train adversarial model
            if model.c.filter and mask.sum() > 0:
                call_fw_bw(
                    settings,
                    x,
                    s,
                    epoch_logs,
                    step=run_step,
                    discriminate=True,
                    batch_storage=batch_storage,
                    mask=mask,
                )

            # Reset virtual batch
            if run_step:
                virtual_counter = 1
                batch_storage = EvalStorage()

        # EVALUATION
        if settings.train_eval and epoch % 10 == 0:
            # Set to evaluation mode
            model.eval()

            if model.c.filter:
                evaluate_masks(settings, epoch_logs, info_align=False)
            else:
                evaluate_model(
                    settings,
                    epoch_logs,
                    eval_representations=eval_representations,
                )

        # Update wand
        if model.c.verbose:
            wandb.log(epoch_logs)

        # Early stopping
        if early_stopping:
            for i, (x, _, s, _) in enumerate(settings.val_loader):
                # Move batch to GPU if Csr is used
                if settings.csr:
                    x = x.to(model.c.device)
                    s = s.to(device=model.c.device, dtype=torch.float32)

                _, _, _, loss = call_fw_bw(settings, x, s, {}, backwards=False)
            stop, es_counter, best_loss, best_model = update_early_stopping(
                loss, best_loss, es_counter, model, best_model, n_es, eps=es_eps
            )
            if stop:
                model.load_state_dict(best_model)
                break

        epoch += 1


def train_adv(settings, epoch_logs, add_noise=False, perturb=False):
    model = settings.model
    device = model.c.device
    # Add noise to the user input when training adversarials for Perturb models.
    # This allows us to learn signals from items that would otherwise be overshadowed by
    # "stronger" items and the dropout itself simulates the removal operation in
    # the Perturb models.
    old_noise_setting = model.add_noise
    model.add_noise = add_noise
    loader = settings.train_loader
    if perturb:
        perturbed_xs = None
        inds = None
        for i, (x, _, s, ind) in enumerate(settings.train_loader):
            if settings.csr:
                x = x.to(device=device)
                s = s.to(device=device, dtype=torch.float32)
            _, _, perturbed_x = perturb_users(settings, x, s, {}, half_budget=True)
            if settings.csr:
                perturbed_x = perturbed_x.detach().cpu().numpy()
                perturbed_x = csr_matrix(perturbed_x)
                perturbed_xs = vstack((perturbed_xs, perturbed_x)) if perturbed_xs is not None else perturbed_x
            else:
                perturbed_xs = torch.vstack((perturbed_xs, perturbed_x)) if perturbed_xs is not None else perturbed_x
            inds = np.vstack((inds, ind)) if inds is not None else ind
        # Index s with shuffled training indices
        inds = inds.reshape((-1))
        s = settings.train_loader.dataset.s[inds]
        y = None
        loader = init_dataloader(perturbed_xs, y, s, device, loader.batch_size, csr=settings.csr, shuffle=True)

    early_stopping = False
    if settings.early_stopping > 0:
        n_es = settings.early_stopping
        early_stopping = True
        best_loss = np.inf
        best_model = None
        es_counter = 0
        es_eps = 1e-5
    upper_epoch = settings.n_adv_pre if not early_stopping else 1000
    epoch = 0
    while epoch < upper_epoch:
        batch_storage = EvalStorage()
        virtual_counter = 1
        for i, (x, _, s, ind) in enumerate(loader):
            # Move batch to GPU if Csr is used
            if settings.csr:
                x = x.to(device=device)
                s = s.to(device=device, dtype=torch.float32)
            if virtual_counter < settings.nbatch_per_update:
                run_step = False
                virtual_counter += 1
            else:
                run_step = True

            z, _, batch_storage, _ = call_fw_bw(
                settings,
                x,
                s,
                epoch_logs,
                step=run_step,
                discriminate=True,
                adv2=perturb,
                batch_storage=batch_storage,
            )

            # Reset virtual batch
            if run_step:
                virtual_counter = 1
                batch_storage = EvalStorage()
        if early_stopping or settings.debug:
            for i, (x, _, s, _) in enumerate(settings.val_loader):
                if settings.csr:
                    x = x.to(device=device)
                    s = s.to(device=device, dtype=torch.float32)
                latent, _, _ = model(x, s, {}, decode=False)
                latent = latent.detach()

                loss = model.discriminate_simple(latent, s, adv2=perturb)

                if settings.debug:
                    print(f"{epoch}: {loss.item()}")
            if early_stopping:
                stop, es_counter, best_loss, best_model = update_early_stopping(
                    loss, best_loss, es_counter, model, best_model, n_es, eps=es_eps
                )
                if stop:
                    model.load_state_dict(best_model)
                    break
        epoch += 1
    if settings.debug:
        print("Pre-training Done")
    model.add_noise = old_noise_setting

    # Find classifier threshold to maximize accuracy
    for i, (x, _, s, _) in enumerate(settings.val_loader):
        if settings.csr:
            x = x.to(device=device)
            s = s.to(device=device, dtype=torch.float32)
        latent, _, _ = model(x, s, {}, decode=False)
        latent = latent.detach()

        _, probs = model.discriminate_simple(latent, s, return_probs=True, adv2=perturb)
        n_tests = 1000
        n_users = probs.shape[0]
        thresholds = []
        b_thresholds = []
        for i in range(model.c.n_sensitive):
            s_i = s[:, i]
            n_1 = s_i.sum()
            n_0 = n_users - n_1
            probs_i = probs[:, i]
            accuracies = np.zeros(n_tests)
            b_accuracies = np.zeros(n_tests)
            for j in range(n_tests):
                threshold = j / n_tests
                classification = probs_i > threshold
                # Balanced accuracy (summed accuracy of class 0 and 1 devided by 2). Interpretation: classifying 50% of males as females is
                # equally bad/good as classifying 50% av females as males. One correct female (minority) classification is worth more than
                # one correct male (majority) classification. Not perfect, but avoids selecting a threshold where most classified as males
                accuracy = (classification == (s_i == 1)).sum() / n_users
                balanced_accuracy = (
                    (classification[s_i == 0] == 0).sum() + (n_0 / n_1) * ((classification[s_i == 1] == 1).sum())
                ) / (2 * n_0)
                accuracies[j] = accuracy.item()
                b_accuracies[j] = balanced_accuracy.item()
            thresholds.append(np.argmax(accuracies) / n_tests)
            b_thresholds.append(np.argmax(b_accuracies) / n_tests)
    return np.array(b_thresholds)


def call_fw_bw(
    settings,
    x,
    s,
    wandb_log,
    backwards=True,
    step=True,
    discriminate=False,
    adv2=False,
    batch_storage=None,
    mask=None,
):
    model = settings.model
    # Skip decoding when training adversarials
    decode = not discriminate

    latent, loss, probs = model(x, s, wandb_log, decode=decode, mask=mask)
    latent = latent.detach()

    # For evaluation purposes
    if not backwards and discriminate and model.c.verbose:
        loss = model.discriminate_simple(latent, s, wandb_log=wandb_log)

    if backwards and discriminate:
        # In case we have multiple virtual batches we have to store latents from different parts
        batch_storage.update(zs=latent, s=s)
        if step:
            # Multiple iterations of training
            batch_latents = torch.cat(batch_storage.zs)
            batch_s = torch.cat(batch_storage.s)
            for i in range(settings.n_adv_train):
                loss = model.discriminate_simple(
                    batch_latents, batch_s, loss_weight=mask, adv2=adv2, wandb_log=wandb_log
                )

                settings.opt_adv.zero_grad()
                loss.backward()
                settings.opt_adv.step()
    elif backwards:
        loss.backward()
        if step:
            settings.opt_model.step()
            settings.opt_model.zero_grad()

    return latent, probs, batch_storage, loss.detach().cpu().numpy()


def evaluate_model(
    settings,
    wandb_log,
    eval_representations=False,
    perturb=False,
):
    model = settings.model
    # Store sample setting and set flag to False
    sample_setting = model.sample
    model.sample = False
    n_sensitive = model.c.n_sensitive
    eval_storage, eval_res = evaluate_model_recs(settings, wandb_log)
    if perturb:

        perturb_vals = np.arange(n_sensitive + 1) - 1 if n_sensitive > 1 else [0]
        perturb_outputs = []
        for perturb_val in perturb_vals:
            perturb_outputs.append(evaluate_model_recs(settings, wandb_log, perturb=perturb_val))

    # Log recommendation performance
    prefixes = ["1" * n_sensitive] if n_sensitive > 1 else []
    for i in range(n_sensitive):
        prefix = ""
        for j in range(n_sensitive):
            if j == i:
                prefix += "1"
            else:
                prefix += "0"
        prefixes.append(prefix)
    if model.c.verbose:
        test_prefix = "test " if perturb else ""
        log_metrics(eval_res, test_prefix, model.c.sensitive_labels, wandb_log)
        if perturb:
            for i, tup in enumerate(perturb_outputs):
                log_metrics(tup[1], prefixes[i], model.c.sensitive_labels, wandb_log)

    # Representation neutrality evaluation
    # Sampling is set to False at this point. Will be reverted through the sample_setting variable later
    if eval_representations:
        main_prefix = "00 " if perturb else "val"
        evaluate_representation(eval_storage.zs, eval_storage.s, main_prefix, model, wandb_log)
        if perturb:
            for i, tup in enumerate(perturb_outputs):
                evaluate_representation(tup[0].zs, eval_storage.s, prefixes[i], model, wandb_log)
            # Example of complex AUC evalution where some users opt to hide different parts
            # zp = [tup[0].zs for tup in perturb_outputs]
            # logreg_complex = logreg_training(20, 5, eval_storage.zs, eval_storage.s, zp=zp, frac=0.15)

    # Revert sampling setting
    model.sample = sample_setting


def evaluate_masks(settings, wandb_log, info_align=True):
    verbose = False
    model = settings.model
    model.eval()
    device = model.c.device
    n_sensitive = settings.model.c.n_sensitive
    masks = [[0] * n_sensitive for i in range(n_sensitive + 1)]
    for i in range(1, n_sensitive + 1):
        masks[i][-i] = 1
    if n_sensitive > 1:
        masks.append([1] * n_sensitive)
    for i, mask in enumerate(masks):
        mask_str = "".join(str(x) for x in mask) + " "
        np_mask = np.array(mask)
        eval_storage = EvalStorage()
        for x, y, s, _ in settings.val_loader:
            if settings.csr:
                x = x.to(device)
                s = s.to(device, dtype=torch.float32)

            # Forward pass
            base_probs = None
            if info_align:
                latent, _, base_probs = model(x, s, {}, decode=True)
                base_probs[x.nonzero(as_tuple=True)] = 0.0

                combined, _ = info_align_encode(model, s, latent, device, np_mask)

                latent = model.combine_net(combined.detach())
                probs, _ = model.decode(x, latent)
            else:
                latent, _, probs = model(x, s, {}, mask=np_mask)
            if sum(mask) == 2 and verbose:
                u_probs = probs.detach().clone()
                u_probs[x.nonzero(as_tuple=True)] = 0.0
                dump_torch_ratio(
                    u_probs,
                    s,
                    settings.k,
                    settings.demographics,
                    settings.processed_dir,
                    ranked=False,
                    old_scores=base_probs,
                )

            eval_storage.process_and_update(probs, x, y, latent, s, False)
        eval_storage.concat()
        eval_res = evaluate_all_recommendations(settings, eval_storage.u_probs, eval_storage.targets)
        log_metrics(eval_res, mask_str, model.c.sensitive_labels, wandb_log)
        aucs = evaluate_representation(eval_storage.zs, eval_storage.s, mask_str, model, wandb_log)


def evaluate_model_recs(settings, wandb_log, perturb=None, mask=None):
    # All probs are stored to perform single evaluation
    if perturb is not None:
        eval_storage = perturb_model(settings, wandb_log, single_sens=perturb)
    else:
        eval_storage, _ = inference(settings, wandb_log, mask=mask)
    eval_res = evaluate_all_recommendations(settings, eval_storage.u_probs, eval_storage.targets)
    return eval_storage, eval_res


def dump_probs(settings, loaders, test_ind, perturb=False, info_align=False, filter=False):
    model = settings.model
    model.eval()
    perturb_settings = settings.perturb_settings
    device = model.c.device
    n_sensitive = settings.model.c.n_sensitive
    mask = np.ones(n_sensitive)
    base_probs = []
    fair_probs = []
    base_auc = -1
    fair_auc = -1
    for i, loader in enumerate(loaders):
        base_storage = EvalStorage()
        fair_storage = EvalStorage()
        for x, y, s, _ in loader:
            if settings.csr:
                x = x.to(device)
                s = s.to(device, dtype=torch.float32)

            # Forward pass
            base_latent, _, base_prob = model(x, s, {}, decode=True)
            if info_align:
                combined, _ = info_align_encode(model, s, base_latent, device, mask)
                latent = model.combine_net(combined.detach())
                fair_prob, _ = model.decode(x, latent)
            elif perturb:
                half = perturb_settings.double_adv
                double = perturb_settings.double_adv
                latent, fair_prob, _ = perturb_users(
                    settings, x, s, {}, single_sens=-1, half_budget=half, double=double
                )
            else:
                latent, _, fair_prob = model(x, s, {}, mask=mask)
            base_storage.process_and_update(base_prob, x, y, base_latent, s, False)
            fair_storage.process_and_update(fair_prob, x, y, latent, s, False)
        base_storage.concat()
        fair_storage.concat()
        base_probs.append(base_storage.u_probs)
        fair_probs.append(fair_storage.u_probs)
        if i == test_ind:
            base_auc = logreg_training(20, 5, base_storage.zs, base_storage.s)
            fair_auc = logreg_training(20, 5, fair_storage.zs, fair_storage.s)
    # Return last eval storage, i.e., the one where all sensitive attributes have been masked
    return base_probs, fair_probs, base_auc, fair_auc


def log_metrics(eval_res, test_prefix, sensitive_labels, wandb_log):
    wandb_log[f"metric/{test_prefix}NDCG"] = eval_res.ndcg


def inference(
    settings,
    wandb_log,
    latent_only=False,
    train_set=False,
    eval_res=None,
    mask=None,
):
    model = settings.model
    # Structures for assembling the correct evaluation data with respect to mini-batching
    eval_storage = EvalStorage()
    loader = settings.train_loader if train_set else settings.val_loader
    for i, (x, y, s, _) in enumerate(loader):
        # Move batch to GPU if Csr is used
        if settings.csr:
            x = x.to(model.c.device)
            s = s.to(device=model.c.device, dtype=torch.float32)

        # Forward pass
        z, probs, _, _ = call_fw_bw(settings, x, s, wandb_log, backwards=False, mask=mask)
        eval_storage.process_and_update(probs, x, y, z, s, latent_only)

    # Assemble evaluation
    eval_storage.concat()
    return eval_storage, eval_res


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
            unique, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            if min_count < 3 or unique.shape[0] < 2:
                raise Exception(
                    "Representation AUC cannot be calculated: too few minority labels in test set. Try with bigger datasets or less skewed classes"
                )
            elif min_count < k_split and min_count >= 3:
                k_split = min_count
            # Perform random split of data
            kf = StratifiedKFold(n_splits=k_split, shuffle=True)

            for train_index, val_index in kf.split(z, y):
                train_z, train_y = z[train_index], y[train_index]
                val_z, val_y = z_val[val_index], y[val_index]

                # Fit and evaluate model
                logreg_model = LogisticRegression(max_iter=200)
                logreg_model.fit(train_z, train_y)
                logreg_probs = logreg_model.predict_proba(val_z)
                logreg[j].append(roc_auc_score(val_y, logreg_probs[:, 1]))

    logreg = [np.mean(logre) for logre in logreg]

    return logreg


if __name__ == "__main__":
    parse_arguments()
