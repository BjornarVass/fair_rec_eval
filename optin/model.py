import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, field
from typing import Any


# Base class with common functionality used in other models
class BaseModel(nn.Module):
    def __init__(self, settings, info_align=False, double_adv=False):
        super().__init__()
        self.c = settings

        # Epsilon for log expressions
        self.eps = 1e-7

        # Add adversarial models
        if self.c.adv:
            adversarials = []
            for i in range(self.c.n_sensitive):
                if self.c.deep_adv:
                    adversarials.append(self.init_deep_adv())
                else:
                    adversarials.append(
                        self.setup_layer(
                            self.c.adv_dims,
                            self.c.hidden_activation,
                            last_activation="sigmoid",
                            dropout=self.c.dropout_adv,
                        )
                    )
                adversarials[i].apply(self.init_weights)
            self.adversarials = nn.ModuleList(adversarials)
            if double_adv:
                adversarials = []
                for i in range(self.c.n_sensitive):
                    adversarials.append(
                        self.setup_layer(
                            self.c.adv_dims,
                            self.c.hidden_activation,
                            last_activation="sigmoid",
                            dropout=self.c.dropout_adv,
                        )
                    )
                    adversarials[i].apply(self.init_weights)
                self.adversarials2 = nn.ModuleList(adversarials)


        # Add filter modules
        # Used in VAEsm, architecture taken from src/model/BaseRecModel.py in:
        # https://github.com/yunqi-li/Personalized-Counterfactual-Fairness-in-Recommendation
        if self.c.filter:
            self.n_filters = self.c.n_sensitive**2 - 1
            filters = []
            for i in range(self.n_filters):
                filter = nn.Sequential(
                    nn.Linear(self.c.latent_dim, self.c.latent_dim * 2),
                    nn.LeakyReLU(),
                    nn.Linear(self.c.latent_dim * 2, self.c.latent_dim),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(self.c.latent_dim)
                )
                filters.append(filter)
                filters[i].apply(self.init_weights)
            self.filters = nn.ModuleList(filters)
        
        # Information alignment
        if info_align:
            # We replaced the attribute encoders in the official implementation with attribute embeddings
            # since we observed that the encoders were trained to perfectly predict/memorize all attributes
            # of all users. There is a possibility that the attribute encodings can produce better representations
            # for the combination step, but we have no reason to believe so and our testing did not challenge this
            # assumption. We have kept the code for the encoder version, but this has not been kept up to date.
            self.sim_info_align = True
            z_dim = self.c.latent_dim
            encode_dims = [z_dim, 2*z_dim, 4*z_dim, 8*z_dim, 4*z_dim, 2*z_dim,z_dim]
            if self.sim_info_align:
                comb_dim = z_dim + self.c.n_sensitive*3
                combine_dims = [comb_dim, 2*comb_dim, comb_dim, z_dim]
            else:
                combine_dims = [3*z_dim, 6*z_dim, 3*z_dim, z_dim]

            align_encode = []
            n_encoders = self.c.n_sensitive+1 if not self.sim_info_align else 1
            for i in range(n_encoders):
                align_encode.append(self.setup_layer(encode_dims, "relu", layer_norm=True))
                align_encode[i].apply(self.init_weights)
            self.align_encode = nn.ModuleList(align_encode)
            
            self.combine_net = self.setup_layer(combine_dims, "relu")
            self.combine_net.apply(self.init_weights)

            if not self.sim_info_align:
                classifiers = []
                for i in range(self.c.n_sensitive):
                    classifiers.append(
                        self.setup_layer(
                            self.c.adv_dims,
                            "relu",
                            last_activation="sigmoid",
                            dropout=self.c.dropout_adv,
                        )
                    )
                    classifiers[i].apply(self.init_weights)
                self.classifiers = nn.ModuleList(classifiers)

    def apply_filter(self, representations, filter_mask):
        # We only implement the separate filter mode
        if np.sum(filter_mask) != 0:
            # filter_mask = np.asarray(filter_mask)
            filter_ind = filter_mask.dot(2**np.arange(filter_mask.size)) - 1
            sens_filter = self.filters[filter_ind]
            representations = sens_filter(representations)
        return representations

    def get_activation(self, hidden_activation):
        # Activation
        if hidden_activation == "tanh":
            return nn.Tanh()
        elif hidden_activation == "relu":
            return nn.ReLU()
        elif hidden_activation == "selu":
            return nn.SELU()
        elif hidden_activation == "gelu":
            return nn.GELU()
        elif hidden_activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f'Unknown hidden activation: "{hidden_activation}"')
            return

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)
            # TODO MF vs VAE
            #torch.nn.init.normal_(m.weight, mean=0.0, std=0.001)
            #if m.bias is not None:
            #    torch.nn.init.normal_(m.bias, mean=0.0, std=0.001)

    def setup_layer(
        self,
        dims,
        hidden_activation,
        last_activation="",
        last_dropout=False,
        dropout=0.0,
        layer_norm=False,
    ):
        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(dims[0]))
        n_dims = len(dims)
        n_layers = n_dims - 1
        for i in range(n_dims - 1):
            from_dim = dims[i]
            to_dim = dims[i + 1]
            layers.append(nn.Linear(from_dim, to_dim))

            if i != n_layers - 1:
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(self.get_activation(hidden_activation))
        if last_dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        if last_activation != "":
            # For some parts, we want an activation after the final layer
            layers.append(self.get_activation(last_activation))
        return nn.Sequential(*layers)
    
    def discriminate_simple(
        self, representation, targets, return_probs=False, loss_weight=None, adv=True, adv2=False, wandb_log=None
    ):
        models = self.adversarials if adv else self.classifiers
        models = self.adversarials2 if adv2 else models
        cross_entropy_loss = 0
        all_probs = torch.empty(
            (representation.shape[0], 0), device=self.c.device, dtype=torch.float32
        )
        for i in range(self.c.n_sensitive):
            # Save computation
            if loss_weight is not None and loss_weight[i] == 0.0:
                continue
            target = targets[:, i]
            probs = models[i](representation)

            all_probs = torch.concat((all_probs, probs), dim=1)

            # Loss
            new_loss_part = torch.matmul(
                target, torch.log(probs + self.eps)
            ) + torch.matmul(1 - target, torch.log((1 - probs) + self.eps))
            if loss_weight is not None:
                new_loss_part *= loss_weight[i]
            cross_entropy_loss += new_loss_part / probs.shape[0]
        cross_entropy_loss = -cross_entropy_loss
        if return_probs:
            return cross_entropy_loss, all_probs
        return cross_entropy_loss
    
    # Used in VAEsm, architecture taken from src/model/Discriminators.py in:
    # https://github.com/yunqi-li/Personalized-Counterfactual-Fairness-in-Recommendation
    def init_deep_adv(self):
        latent_dim = self.c.latent_dim
        neg_slope = 0.2 # Hard-coded. Source code use this default value and do not override in experiments
        adversarial = nn.Sequential(
            nn.Linear(latent_dim, int(latent_dim * 2)),
            # nn.BatchNorm1d(num_features=latent_dim * 4),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=self.c.dropout_adv),
            nn.Linear(latent_dim * 2, latent_dim * 4),
            # nn.BatchNorm1d(num_features=latent_dim * 2),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=self.c.dropout_adv),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            # nn.BatchNorm1d(num_features=latent_dim),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=self.c.dropout_adv),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=self.c.dropout_adv),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=self.c.dropout_adv),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(neg_slope),
            nn.Dropout(p=self.c.dropout_adv),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        return adversarial


class FairVAERec(BaseModel):
    def __init__(self, settings, info_align=False, double_adv=False):
        super().__init__(settings, info_align, double_adv)

        # Derived flags
        self.z_dim = self.c.z_enc_dims[-1] // 2

        # Flag for sampling in eval mode (drop out will not be enabled)
        self.sample = False

        # Flag for adding noise to input data in eval mode
        self.add_noise = False

        # Init loss funcs
        self.bce_loss = nn.BCELoss()

        # Add z encoder
        self.z_encoder = self.setup_layer(
            self.c.z_enc_dims, self.c.hidden_activation, dropout=self.c.dropout
        )
        self.z_encoder.apply(self.init_weights)

        # Add z decoder
        self.z_decoder = self.setup_layer(
            self.c.z_dec_dims,
            self.c.hidden_activation,
            dropout=self.c.dropout,
        )
        self.z_decoder.apply(self.init_weights)


    def get_params(self, adversarial=False):
        if not adversarial:
            params = list(self.z_encoder.parameters()) + list(
                self.z_decoder.parameters()
            )
            if self.c.filter:
                for filter in self.filters:
                    params = params + list(filter.parameters())
        else:
            params = []
            for adv in self.adversarials:
                params = params + list(adv.parameters())
        return params

    def encode(self, x):
        # Normalize to remedy issues related to active vs passive users
        h = F.normalize(x, dim=1)
        # Add noise, noisy VAE
        if self.training or self.add_noise:
            h = F.dropout(h, p=self.c.noise_rate)

        # Encode
        h_z = self.z_encoder(h)

        # Extract mu and log(sigma^2) from h_z
        mu = h_z[:, : self.z_dim]
        logvar = h_z[:, self.z_dim :]

        # Calculate standard deviation and KL divergence
        std = torch.exp(0.5 * logvar)
        KL = torch.mean(
            torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu**2 - 1), dim=1)
        )

        return mu, std, KL

    def forward(self, x, s, wandb_log, decode=True, mask=None):
        # Encode and calc KL divergence
        mu, std, KL = self.encode(x)

        # Sample latents
        sampled_z = self.reparam_trick(mu, std)

        # Apply masks
        filter_flag = self.c.filter and mask is not None and mask.sum() > 0
        if filter_flag:
            sampled_z = self.apply_filter(sampled_z, mask)

        # Skip decoding
        if not decode:
            return sampled_z, None, None

        # x reconstruct term
        probs, negative_ll = self.decode(x, sampled_z)

        # Shared loss parts and logging
        neg_elbo = negative_ll + self.c.beta * KL

        prefix = "train" if self.training else "validation"
        if self.c.verbose:
            wandb_log[f"{prefix}/rec recon loss"] = negative_ll
            wandb_log[f"{prefix}/KL loss"] = self.c.beta * KL

        # Add additional loss terms
        if filter_flag:
            adv_loss = self.discriminate_simple(sampled_z, s, loss_weight=mask)

            full_loss = neg_elbo - self.c.gamma * adv_loss
        else:
            full_loss = neg_elbo

        if self.c.verbose:
            wandb_log[f"{prefix}/loss"] = full_loss
        return sampled_z, full_loss, probs
    

    def decode(self, x, z, return_logits=False):
        logits = self.z_decoder(z)
        log_softmax = F.log_softmax(logits, dim=1)
        negative_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))
        if not return_logits:
            return F.softmax(logits, dim=1), negative_ll
        else:
            return logits, negative_ll


    def discriminate_single(self, z, targets, sensitive_index, return_probs=False):
        cross_entropy_loss = 0
        target = targets[:, sensitive_index]
        probs = self.adversarials[sensitive_index](z)

        # Loss
        loss = torch.matmul(target, torch.log(probs + self.eps)) + torch.matmul(
            1 - target, torch.log((1 - probs) + self.eps)
        )
        cross_entropy_loss = -loss
        if return_probs:
            return cross_entropy_loss, probs
        return cross_entropy_loss

    def reparam_trick(self, mu, std):
        epsilon = torch.empty(
            std.shape, dtype=torch.float32, device=self.c.device
        ).normal_(mean=0, std=1)
        # Turn on sampling during training or whenever sampling is turned on
        sample_flag = self.training or self.sample
        sampled = mu + sample_flag * epsilon * std
        return sampled
    

class WRMF(BaseModel):
    def __init__(self, settings):
        super().__init__(settings)
        self.c = settings
        self.drop = nn.Dropout(self.c.dropout)

        self.user_embedding = torch.nn.Embedding(self.c.n_users, self.c.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.c.n_items, self.c.latent_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        self.user_bias = torch.nn.Embedding(self.c.n_users, 1)
        self.user_bias.weight.data = torch.zeros(self.c.n_users, 1).float()
        self.item_bias = torch.nn.Embedding(self.c.n_items, 1)
        self.item_bias.weight.data = torch.zeros(self.c.n_items, 1).float()


    def get_params(self, adversarial=False):
        if not adversarial:
            params = (
                list(self.user_embedding.parameters())
                + list(self.item_embedding.parameters())
                + list(self.user_bias.parameters())
                + list(self.item_bias.parameters())
            )
        else:
            params = []
            for adv in self.adversarials:
                params = params + list(adv.parameters())
        return params

    def predict(self, user_indices, item_indices, user_embedding=None):
        if user_embedding is not None:
            user_vec = user_embedding
        else:
            user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        if self.c.dropout > 0:
            user_vec = self.drop(user_vec)
            item_vec = self.drop(item_vec)
        dot = (user_vec * item_vec).sum(2)
        u_bias = self.user_bias(user_indices).view(user_indices.shape)
        i_bias = self.item_bias(item_indices).view(item_indices.shape)
        score = dot + i_bias  + u_bias

        return score

    def forward(self, x, user_indices, item_indices):
        score = self.predict(user_indices, item_indices)
        l2 = self.c.reg * (
            torch.sum(self.user_embedding(user_indices) ** 2)
            + torch.sum(self.item_embedding(item_indices) ** 2)
        )
        loss = l2 + torch.mean((x * 0.5 + 0.5) * (score - x) ** 2)
        return loss

@dataclass
class ModelSettings(object):
    # VAE specific
    beta: float = 1.0
    noise_rate: float = 0.5
    z_enc_dims: list[int] = field(default_factory=list)
    z_dec_dims: list[int] = field(default_factory=list)
    # MF specific
    reg: float = 0.0
    n_users: int = 0
    n_items: int = 0
    # Common
    latent_dim: int = 64
    adv: bool = False
    deep_adv: bool = False
    adv_thresholds: Any = None
    adv_thresholds2: Any = None
    adv_dims: list[int] = field(default_factory=list)
    dropout: float = 0.0
    dropout_adv: float = 0.0
    n_sensitive: int = 0
    sensitive_labels: list[str] = field(default_factory=list)
    hidden_activation: str = "tanh"
    device: str = "cuda:0"
    verbose: bool = False
    # VAEsm
    filter: bool = False
    gamma: float = 0.0

@dataclass
class RunSettings(object):
    model: Any = None
    opt_model: Any = None
    opt_adv: Any = None
    train_loader: Any = None
    val_loader: Any = None
    n_epochs: int = 0
    n_adv_pre: int = 0
    nbatch_per_update: int = 1
    discriminate: bool = False
    n_adv_train: int = 1
    k: int = 10
    train_eval: bool = True
    rep_eval_interval: int = 100
    csr: bool = False
    title_map: Any = None
    processed_dir: Any = None
    demographics: Any = None
    debug: bool = False
    early_stopping: int = -1
    # VAEafrl settings
    info_align: bool = False
    afrl: Any = None
    # Perturb
    perturb: bool = False
    perturb_settings: Any = None

@dataclass
class VAEafrlSettings(object):
    beta: float = 0.1
    lambd: float = 1.0
    weight_decay: float = 1e-8
    lr: float = 5e-5
    n_encode_epochs: int = 1000
    n_combine_epochs: int = 500

@dataclass
class PerturbSettings(object):
    grad_only: bool = False
    double_adv: bool = False
    max_budget: int = 50
    adv_prob_threshold: float = 0.025

@dataclass
class BudgetStruct(object):
    top_2k: dict = None
    removal_cands: set = None
    added_recs: list = None
    taboo_set: set = None
    added: list = None
    removed: list = None
    n_orig_items: int = 0
    n_current: int = 0
    n_max_rem: int = 0
    n_max_add: int = 0


class EvalStorage(object):
    def __init__(self):
        self.zs = []
        self.s = []
        self.probs = []
        self.u_probs = []
        self.targets = []
        # Training reps and s for training aux models (AUC)
        self.z_tr = []
        self.s_tr = []

    def update(self, zs=None, s=None, probs=None, u_probs=None, targets=None, z_tr=None, s_tr=None):
        if zs is not None:
            self.zs.append(zs)
        if s is not None:
            self.s.append(s)
        if probs is not None:
            self.probs.append(probs)
        if u_probs is not None:
            self.u_probs.append(u_probs)
        if targets is not None:
            self.targets.append(targets)
        if z_tr is not None:
            self.z_tr.append(z_tr)
        if s_tr is not None:
            self.s_tr.append(s_tr)

    def process_and_update(self, probs, x, y, z, s, latent_only):
        z_np = z.cpu().detach().numpy()
        s_np = s.cpu().detach().numpy()

        if latent_only:
            self.update(zs=z_np, s=s_np)
            return

        # Move probs to CPU for evaluation
        probs = probs.cpu().detach().numpy()
        # Ignore items we know the user likes in NDCG scores
        util_probs = np.array(probs)
        x_np = x.cpu().detach().numpy()
        util_probs[x_np.nonzero()] = -np.inf

        self.update(zs=z_np, s=s_np, probs=probs, u_probs=util_probs, targets=y.detach().numpy())

    def concat(self):
        self.zs = np.concatenate(self.zs)
        self.s = np.concatenate(self.s)
        self.probs = np.concatenate(self.probs) if self.probs != [] else []
        self.u_probs = np.concatenate(self.u_probs) if self.u_probs != [] else []
        self.targets = np.concatenate(self.targets) if self.targets != [] else []
        self.z_tr = np.concatenate(self.z_tr) if self.z_tr != [] else []
        self.s_tr = np.concatenate(self.s_tr) if self.s_tr != [] else []


class EvalResults(object):
    def __init__(self, n_sensitive):
        self.n_sensitive = n_sensitive
        self.ndcg = 0

    def set_results(self, ndcg):
        if ndcg is not None:
            self.ndcg = ndcg

    def update(self, ndcg, batch_size):
        self.ndcg += ndcg * batch_size

    def aggregate_parts(self, n_users):
        self.ndcg = self.ndcg / n_users
