"""Stage 8: Component processing for the tedana workflow.

This stage handles:
- Saving ICA mixing matrix
- Computing and saving z-scored component maps
- Calculating rejected components impact metric
- Saving component selector outputs
- Saving metric metadata
- Optional TEDORT orthogonalization
"""

import logging

import numpy as np
import pandas as pd

from tedana import io, metrics, reporting
from tedana.stats import computefeats2
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_component_processing(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Process and save component-related outputs.

    This stage saves the ICA mixing matrix, computes z-scored
    component maps, calculates the rejected components impact
    metric, saves component selector outputs and metadata,
    and optionally applies TEDORT orthogonalization.

    Parameters
    ----------
    config : WorkflowConfig
        Immutable workflow configuration.
    state : WorkflowState
        Mutable workflow state (modified in place).

    Returns
    -------
    StageResult
        Execution result with success/failure status.
    """
    # Save ICA mixing matrix
    comp_names = state.component_table["Component"].values
    mixing_df = pd.DataFrame(data=state.mixing, columns=comp_names)
    state.io_generator.save_file(mixing_df, "ICA mixing tsv")

    # Compute and save z-scored component maps
    betas_oc = computefeats2(state.data_optcom, state.mixing, state.mask_denoise)
    betas_oc = io.unmask(betas_oc, state.mask_denoise)
    state.io_generator.save_file(betas_oc, "z-scored ICA components img")

    # Calculate the fit of rejected to accepted components as a quality measure
    # Note: This adds a column to component_table & needs to run before the table is saved
    reporting.quality_metrics.calculate_rejected_components_impact(state.selector, state.mixing)

    # Save component selector and tree
    state.selector.to_files(state.io_generator)

    # Save metrics metadata
    metric_metadata = metrics.collect.get_metadata(state.component_table)
    state.io_generator.save_file(metric_metadata, "ICA metrics json")

    # Save decomposition metadata
    decomp_metadata = {
        "Method": (
            "Independent components analysis with FastICA algorithm implemented by sklearn. "
        ),
    }
    for comp_name in comp_names:
        decomp_metadata[comp_name] = {
            "Description": "ICA fit to dimensionally-reduced optimally combined data.",
            "Method": "tedana",
        }
    state.io_generator.save_file(decomp_metadata, "ICA decomposition json")

    # Apply TEDORT orthogonalization if requested
    state.mixing_orig = state.mixing.copy()
    if config.tedort:
        _apply_tedort(state)

    return StageResult.ok()


def _apply_tedort(state: WorkflowState) -> None:
    """Apply TEDORT orthogonalization to mixing matrix.

    This orthogonalizes rejected components' time series with respect
    to accepted components' time series.

    Parameters
    ----------
    state : WorkflowState
        Workflow state (modified in place).
    """
    comps_accepted = state.selector.accepted_comps_
    comps_rejected = state.selector.rejected_comps_

    acc_ts = state.mixing[:, comps_accepted]
    rej_ts = state.mixing[:, comps_rejected]

    # Orthogonalize rejected w.r.t. accepted
    betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
    pred_rej_ts = np.dot(acc_ts, betas)
    resid = rej_ts - pred_rej_ts
    state.mixing[:, comps_rejected] = resid

    # Save orthogonalized mixing matrix
    comp_names = [
        io.add_decomp_prefix(comp, prefix="ICA", max_value=state.component_table.index.max())
        for comp in range(state.selector.n_comps_)
    ]
    mixing_df = pd.DataFrame(data=state.mixing, columns=comp_names)
    state.io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")

    RepLGR.info(
        "Rejected components' time series were then "
        "orthogonalized with respect to accepted components' time "
        "series."
    )
