import numpy as np
import torch

from optin.model import BudgetStruct, EvalStorage
from optin.utils import dump_torch_ratio


def get_grad_mask(probs, s, neutral_probs, prob_threshold_h, ignore_outliers=False, single_sens=-1):
    mask = np.zeros(s.shape[0])
    norm_l = 1 / neutral_probs
    norm_h = 1 / (1 - neutral_probs)
    prob_threshold_l = prob_threshold_h * (neutral_probs / (1 - neutral_probs))
    for i in range(s.shape[0]):
        if single_sens > -1 and i != single_sens:
            continue
        ref_val = neutral_probs[i] - probs[i]
        high = ref_val < 0.0
        # Scale distance measures if they are confirmed to be outside the scaled threshold
        if high and -ref_val > prob_threshold_h:
            ref_val = ref_val * norm_h[i]
            if ignore_outliers and s[i] == 0:
                ref_val = 0.0
        elif ref_val > prob_threshold_l[i]:
            ref_val = ref_val * norm_l[i]
            if ignore_outliers and s[i] == 1:
                ref_val = 0.0
        else:
            continue
        # Negate distance measure if user is part of minority (since we use one probability instead of
        # one for each binary class)
        if s[i] == 1:
            ref_val = -ref_val
        mask[i] = ref_val
    return mask


def check_budget(budget):
    return len(budget.added) < budget.n_max_add or len(budget.removed) < budget.n_max_rem


def transparency_latex(x_inds_np, out_probs, all_base_recs, title_map):
    out_probs_np = out_probs[0].detach().cpu().numpy()
    out_recs = np.argsort(-out_probs_np)
    rec_comp_str = ""
    base_recs = []
    new_recs = []
    base_str = ""
    new_str = ""
    for i in range(10):
        base_str += f"{i} {str(title_map[all_base_recs[i]])}\n"
        base_recs.append(all_base_recs[i])
        new_str += f"{i} {str(title_map[out_recs[i]])}\n"
        new_recs.append(out_recs[i])

    # Rec change output
    rec_change_str = rec_change(base_recs, new_recs, title_map, 10)

    # New recs and seen movies latex table output
    for i in range(10):
        rec_comp_str += f"{i+1} & {title_map[all_base_recs[i]][3]} & {title_map[out_recs[i]][3]}\\\\\n"
    seen_movies_str = ""
    for i, ind in enumerate(x_inds_np):
        seen_movies_str += (
            f"{title_map[ind][3]} & {title_map[ind][1]:.2f} & {title_map[ind][2]:.2f} & {title_map[ind][0]}\\\\\n"
        )
    return rec_change_str, rec_comp_str, seen_movies_str


# Function used to print transparency information
def rec_change(recs1, recs2, title_map, k):
    out_str = ""
    for i in range(k):
        rec2 = recs2[i]
        chg2 = 0
        for j in range(recs1.shape[0]):
            if rec2 == recs1[j]:
                chg2 = j - i
        out_str += (
            f"{title_map[rec2][3]}&{title_map[rec2][1]:.2f}&{title_map[rec2][2]:.2f}&{chg2}&{title_map[rec2][0]}\\\\\n"
        )
    return out_str


def perturb_model(settings, wandb_log, single_sens=-1):
    model = settings.model
    perturb_settings = settings.perturb_settings
    # Structures for assembling the correct evaluation data with respect to mini-batching
    eval_storage = EvalStorage()

    for i, (x, y, s, _) in enumerate(settings.val_loader):
        # Move batch to GPU if Csr is used
        if settings.csr:
            x = x.to(model.c.device)
            s = s.to(device=model.c.device, dtype=torch.float32)

        # Forward pass
        half = perturb_settings.double_adv
        double = perturb_settings.double_adv
        z, probs, _ = perturb_users(
            settings, x, s, wandb_log, single_sens=single_sens, half_budget=half, double=double
        )
        eval_storage.process_and_update(probs, x, y, z, s, False)

    # Assemble evaluation
    eval_storage.concat()
    return eval_storage


def perturb_users(settings, x, s, wandb_log, single_sens=-1, half_budget=False, double=False):
    model = settings.model
    perturb_settings = settings.perturb_settings

    # Get top k ORIGINAL recs
    base_latent, _, base_probs = model(x, s, wandb_log, decode=True)
    input_indices = x.nonzero(as_tuple=True)
    base_probs[input_indices] = 0.0
    dual_purpose_recs = torch.topk(base_probs, k=2 * settings.k, dim=1, sorted=True)

    # Move to cpu
    sensitive_counts = settings.train_loader.dataset.s.sum(0)
    demographics = sensitive_counts / settings.train_loader.dataset.s.shape[0]
    if not settings.csr:
        demographics = demographics.detach().cpu().numpy()
    base_rec_inds = dual_purpose_recs.indices.detach().cpu().numpy()
    base_rec_probs = dual_purpose_recs.values.detach().cpu().numpy()
    all_base_recs = np.argsort(-base_probs.detach().cpu().numpy(), axis=1)

    adv_thresholds = model.c.adv_thresholds

    # Base adv
    _, base_adv_probs = model.discriminate_simple(base_latent, s, return_probs=True)
    prob_threshold = perturb_settings.adv_prob_threshold

    # List for batch output aggregation
    new_latents = []
    new_probs = []
    new_xs = []
    for i in range(x.shape[0]):
        # Help structs
        x_inds_np = torch.nonzero(x[i]).detach().cpu().numpy()[:, 0]
        top_2k = dict(zip(base_rec_inds[i], base_rec_probs[i]))
        removal_cands = set(x_inds_np.tolist())
        added_recs = []
        taboo_set = set(x_inds_np.tolist())
        added = []
        removed = []

        # Initialize budget
        max_add_rem = perturb_settings.max_budget
        n_orig_items = len(removal_cands)
        n_current = n_orig_items
        n_max_rem = min(n_orig_items - 5, n_orig_items // 2, max_add_rem)
        n_max_add = min(n_orig_items // 2, max_add_rem)
        first_rem = n_max_rem // 2 if half_budget else n_max_rem
        first_add = n_max_add // 2 if half_budget else n_max_add
        budget = BudgetStruct(
            top_2k=top_2k,
            removal_cands=removal_cands,
            added_recs=added_recs,
            taboo_set=taboo_set,
            added=added,
            removed=removed,
            n_orig_items=n_orig_items,
            n_current=n_current,
            n_max_rem=first_rem,
            n_max_add=first_add,
        )

        # Perturb user
        new_latent, new_prob, new_x = perturb_user(
            settings,
            model,
            x[i : i + 1],
            s[i : i + 1],
            base_adv_probs[i].detach().cpu().numpy(),
            adv_thresholds,
            settings.title_map,
            budget,
            base_probs[i],
            prob_threshold,
            wandb_log,
            all_base_recs[i],
            single_sens=single_sens,
        )
        # Perturb2Adv: Update budget and perturb user further with second adversarial group
        if double:
            budget.n_max_rem = n_max_rem
            budget.n_max_add = n_max_add
            adv_thresholds = model.c.adv_thresholds2
            _, n_adv_probs = model.discriminate_simple(new_latent, s[i : i + 1], return_probs=True, adv2=True)
            new_latent, new_prob, new_x = perturb_user(
                settings,
                model,
                new_x,
                s[i : i + 1],
                n_adv_probs[0].detach().cpu().numpy(),
                adv_thresholds,
                settings.title_map,
                budget,
                base_probs[i],
                prob_threshold,
                wandb_log,
                all_base_recs[i],
                single_sens=single_sens,
                second_adv=double,
            )

        new_latents.append(new_latent)
        new_probs.append(new_prob)
        new_xs.append(new_x)
        # new_recs = torch.topk(new_prob, k=settings.k, dim=1, sorted=True)
    new_latents = torch.concat(new_latents, dim=0)
    new_probs = torch.concat(new_probs)
    new_xs = torch.concat(new_xs)

    # Debug: dump ratio data
    if single_sens == -1:
        dump_torch_ratio(
            new_probs,
            s,
            settings.k,
            settings.demographics,
            settings.processed_dir,
            ranked=False,
            old_scores=base_probs,
        )
        dump_torch_ratio(
            new_probs, s, settings.k, settings.demographics, settings.processed_dir, ranked=True, old_scores=base_probs
        )
    return new_latents, new_probs, new_xs


def perturb_user(
    settings,
    model,
    x_orig,
    s_orig,
    base_adv_probs,
    adv_thresh,
    title_map,
    budget,
    base_probs,
    prob_threshold,
    wandb_log,
    all_base_recs,
    single_sens=-1,
    second_adv=False,
):
    # Debug: transparency output
    transparency_print = False
    perturb_settings = settings.perturb_settings

    x = x_orig.detach().clone()
    s = s_orig.detach().clone()
    x_inds_np = torch.nonzero(x_orig).detach().cpu().numpy()[:, 1]

    transparency_output = (
        None
        if not transparency_print
        else f"{budget.n_orig_items} & & & & & & {base_adv_probs[0]:.2f} & {base_adv_probs[1]:.2f}\\\\\n"
    )

    mask = get_grad_mask(base_adv_probs, s[0], adv_thresh, prob_threshold, single_sens=single_sens)
    new_probs = np.zeros(s.shape[1])
    free_budget = check_budget(budget)
    while mask.any() and free_budget:
        x.requires_grad = True

        # Get gradients given current (perturbed) x
        latent, _, _ = model(x, s, wandb_log, decode=False)
        loss = model.discriminate_simple(latent, s, loss_weight=mask, adv2=second_adv)
        loss.backward()

        # Get gradients and define relevance as gradients weighted by base rec probs (if enabled)
        gradients = x.grad
        if perturb_settings.grad_only:
            relevance = x.grad
        else:
            relevance = x.grad * base_probs
        gradients_np = gradients.detach().cpu().numpy()[0]
        relevance_np = relevance.detach().cpu().numpy()[0]

        # Add or remove item
        x, budget, transparency_output = perturb_item(
            x.detach().clone(), gradients_np, relevance_np, budget, title_map, transparency_print, transparency_output
        )

        # Get new latents and adv probs
        latent, _, _ = model(x, s, {}, decode=False)
        _, new_probs = model.discriminate_simple(latent, s, return_probs=True, adv2=second_adv, wandb_log=wandb_log)
        adv_probs_np = new_probs[0].detach().cpu().numpy()

        # Get loss mask
        mask = get_grad_mask(adv_probs_np, s[0], adv_thresh, prob_threshold, single_sens=single_sens)

        # Check remaining budget
        free_budget = len(budget.added) < budget.n_max_add or len(budget.removed) < budget.n_max_rem

        # Transparency output
        if transparency_print:
            transparency_output += f"{adv_probs_np[0]:.2f} & {adv_probs_np[1]:.2f}\\\\\n"

    # Get final latent and rec probs
    out_latent, _, out_probs = model(x, s, wandb_log, decode=True)

    # Zero out probabilities assigned to perturbed and original input
    ignore_indices = x.nonzero(as_tuple=True)
    ignore_indices_orig = x_orig.nonzero(as_tuple=True)
    out_probs[ignore_indices] = 0.0
    out_probs[ignore_indices_orig] = 0.0

    # Copy over base model rec probs for added items that are in the top-k recommendations of the base model
    for i in budget.added_recs:
        # Got "TypeError: can't assign a numpy.float32 to a torch.cuda.FloatTensor" without float cast. Pytorch bug?
        out_probs[0, i] = float(budget.top_2k[i])

    # Debug:
    if transparency_print:
        rec_change_str, rec_comp_str, seen_movies_str = transparency_latex(
            x_inds_np, out_probs, all_base_recs, title_map
        )
        print(transparency_output)
        print(rec_change_str)
        print(rec_comp_str)
        print(seen_movies_str)

    return out_latent, out_probs, x


def perturb_item(next_x, gradients_np, relevance_np, budget, title_map, transparency_print, transparency_output):
    # Decide whether to add or remove
    add = True
    if len(budget.removed) < budget.n_max_rem and (len(budget.added) == budget.n_max_add or np.random.rand(1) < 0.5):
        add = False

    if not add:
        # Removal is only based on gradients since the relevance is misleading for items in the input
        removal_cands_np = np.array(list(budget.removal_cands))
        removal_local_ind = np.argmin(gradients_np[removal_cands_np])
        removal_ind = removal_cands_np[removal_local_ind]

        # Update budget and x
        budget.removal_cands.remove(removal_ind)
        budget.removed.append(removal_ind)
        budget.n_current -= 1

        next_x[0, removal_ind] = 0.0

        # Debug
        if transparency_print:
            transparency_output += f"{budget.n_current} & Rem & {title_map[removal_ind][3]} & {title_map[removal_ind][1]:.2f} & {title_map[removal_ind][2]:.2f} & {title_map[removal_ind][0]} & "
    else:
        # Addition may be influenced by relevance if enabled
        # Try partition
        n_partition = 2 * budget.n_orig_items
        arg_part = np.argpartition(relevance_np, -n_partition)[-n_partition:]
        greatest = -np.inf
        addition_ind = -1
        for cand in arg_part:
            if cand not in budget.taboo_set and relevance_np[cand] > greatest:
                addition_ind = cand
                greatest = relevance_np[cand]
        # Do full sort in cases where we did not find a candidate in the partition
        if addition_ind == -1:
            sorted_args = np.argsort(relevance_np)[::-1]
            for cand in sorted_args:
                if cand not in budget.taboo_set:
                    addition_ind = cand
                    greatest = relevance_np[cand]
                    break

        # Update budget and x
        if addition_ind in budget.top_2k:
            budget.added_recs.append(addition_ind)
        budget.taboo_set.add(addition_ind)
        budget.added.append(addition_ind)
        budget.n_current += 1

        next_x[0, addition_ind] = 1.0

        # Debug
        if transparency_print:
            transparency_output += f"{budget.n_current} & Add & {title_map[addition_ind][3]} & {title_map[addition_ind][1]:.2f} & {title_map[addition_ind][2]:.2f} & {title_map[addition_ind][0]} & "
    return next_x, budget, transparency_output
