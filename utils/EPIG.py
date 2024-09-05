import logging
import math
from torch import Tensor
import torch

######################################################################3
# EPIG_MT
#######################################################################
def check(
    scores: Tensor, max_value: float = math.inf, epsilon: float = 1e-6, score_type: str = ""
) -> Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        
        logging.warning(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")
    
    return scores

def conditional_epig_from_probs(probs_pool, probs_targ):
    """
    See conditional_epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [k, N_p, Cl_p]
        probs_targ: Tensor[float], [k, N_t, Cl_t]

        --> k   = # forward passes
        --> N_p = # pool_set mol
        --> N_t = # test_set mol
        --> Cl_p = # tasks in poolset (12)
        --> Cl_t = # tasks in testset (1)


    Returns:
        Tensor[float], [N_p, N_t, cl_p]
    """

    # Estimate the joint predictive distribution.
    probs_pool = probs_pool[:, :, None, :, None]  # [K, N_p,  1,  Cl_p,   1]
    probs_targ = probs_targ[:, None, :, None, :]  # [K, 1,   N_t,  1,   Cl_t]
    probs_pool_targ_joint = probs_pool * probs_targ  # [K, N_p, N_t, Cl_p, Cl_t]
    probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)  # [N_p, N_t, Cl_p, Cl_t]

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=0)  # [N_p, 1,  Cl_p,  1]
    probs_targ = torch.mean(probs_targ, dim=0)  # [1,  N_t,  1,   Cl_t]

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl_p, Cl_t]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    nonzero_joint = probs_pool_targ_joint > 0  # [N_p, N_t, Cl_p, Cl_t]
    log_term = torch.clone(probs_pool_targ_joint)  # [N_p, N_t, Cl_p, Cl_t]
    log_term[nonzero_joint] = torch.log(probs_pool_targ_joint[nonzero_joint])  # [N_p, N_t, Cl_p, Cl_t]
    log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])  # [N_p, N_t, Cl_p, Cl_t]

    #scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))  # [N_p, N_t]
    scores = torch.sum(probs_pool_targ_joint * log_term, dim=-1)  # [N_p, N_t, Cl_p]
    return scores  # [N_p, N_t, cl_p]

def epig_from_conditional_scores(scores):
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t, cl_p]

    Returns:
        Tensor[float], [N_p,cl_p]
    """
    scores = torch.mean(scores, dim=1)  # [N_p,cl_p]
    scores = check(scores, score_type="EPIG")
    return scores  # [N_p,cl_p]

def EPIG_MT_acquisition_function(probs_pool, probs_targ):
    """
    See epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [K, N_p, Cl_p]
        probs_targ: Tensor[float], [K, N_t, Cl_t]

    Returns:
        Tensor[float], [N_p,cl_t]
    """
    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t, cl_p]
    return epig_from_conditional_scores(scores)  # [N_p,cl_p]
#############################################################################################