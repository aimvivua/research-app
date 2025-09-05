import numpy as np
from scipy.stats import norm

def calculate_sample_size_buderer(se_target, sp_target, prev, ci_half_width, alpha):
    """
    Calculates sample size for a diagnostic accuracy study using Buderer's method.
    n_pos = N of positive subjects, n_neg = N of negative subjects
    """
    z_alpha_2 = norm.ppf(1 - alpha / 2)
    
    n_pos = (z_alpha_2**2 * se_target * (1 - se_target)) / (ci_half_width**2)
    n_neg = (z_alpha_2**2 * sp_target * (1 - sp_target)) / (ci_half_width**2)
    
    return n_pos, n_neg

def calculate_auc_n(auc, ci_half_width):
    """
    Estimates sample size for an AUC with a desired CI half-width.
    Uses a rule-of-thumb formula derived from the standard error of the AUC.
    """
    se_auc_approx = ci_half_width / norm.ppf(0.975)
    
    # A simplified formula for sample size based on SE_AUC
    n = (0.016 / se_auc_approx)**2
    
    return n

def calculate_epv(num_vars, event_rate, epv_rule):
    """
    Calculates required sample size based on Events Per Variable (EPV).
    """
    required_events = num_vars * epv_rule
    total_n = required_events / event_rate
    return required_events, total_n

def calculate_design_effect(m, icc):
    """
    Calculates the design effect for clustered sampling.
    m = average cluster size
    icc = intra-class correlation coefficient
    """
    deff = 1 + (m - 1) * icc
    return deff