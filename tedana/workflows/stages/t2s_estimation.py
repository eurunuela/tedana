"""Stage 3: T2*/S0 estimation for the tedana workflow.

This stage handles:
- Fitting the monoexponential decay model
- Capping T2* values at reasonable limits
- Saving T2* and S0 maps
- Computing RMSE of the fit
"""

import logging

from scipy import stats

from tedana import decay, utils
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_t2s_estimation(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Estimate T2* and S0 maps from multi-echo data.

    This stage fits a monoexponential decay model to the data,
    caps T2* values at reasonable limits, saves the T2* and S0 maps,
    and computes the RMSE of the fit.

    Can be skipped if a pre-computed T2* map is provided.

    Parameters
    ----------
    config : WorkflowConfig
        Immutable workflow configuration.
    state : WorkflowState
        Mutable workflow state (modified in place).

    Returns
    -------
    StageResult
        Execution result; skipped=True if t2smap was provided.
    """
    # Check if T2* map was pre-computed (set in data_loading stage)
    if state.t2s_full is not None:
        return StageResult.skip("Pre-computed T2* map was provided")

    LGR.info("Computing T2* map")

    # Fit monoexponential decay
    t2s_limited, s0_limited, t2s_full, s0_full = decay.fit_decay(
        data=state.data_cat,
        tes=state.tes,
        mask=state.mask_denoise,
        adaptive_mask=state.masksum_denoise,
        fittype=config.fittype,
        n_threads=config.n_threads,
    )

    # Cap T2* values at 10x the 99.5th percentile
    cap_t2s = stats.scoreatpercentile(t2s_full.flatten(), 99.5, interpolation_method="lower")
    LGR.debug(f"Setting cap on T2* map at {utils.millisec2sec(cap_t2s):.5f}s")
    t2s_full[t2s_full > cap_t2s * 10] = cap_t2s

    # Save T2* and S0 maps
    state.io_generator.save_file(utils.millisec2sec(t2s_full), "t2star img")
    state.io_generator.save_file(s0_full, "s0 img")

    # Save limited maps if verbose
    if config.verbose:
        state.io_generator.save_file(utils.millisec2sec(t2s_limited), "limited t2star img")
        state.io_generator.save_file(s0_limited, "limited s0 img")

    # Calculate RMSE of fit
    rmse_map, rmse_df = decay.rmse_of_fit_decay_ts(
        data=state.data_cat,
        tes=state.tes,
        adaptive_mask=state.masksum_denoise,
        t2s=t2s_limited,
        s0=s0_limited,
        fitmode="all",
    )
    state.io_generator.save_file(rmse_map, "rmse img")
    state.io_generator.add_df_to_file(rmse_df, "confounds tsv")

    # Store results in state
    state.t2s_limited = t2s_limited
    state.s0_limited = s0_limited
    state.t2s_full = t2s_full
    state.s0_full = s0_full

    return StageResult.ok()
