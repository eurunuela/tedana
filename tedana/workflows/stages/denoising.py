"""Stage 9: Denoising output for the tedana workflow.

This stage handles:
- Writing denoised time series
- Optional minimum image regression (MIR)
- Optional per-echo denoised data (verbose mode)
"""

import logging

import tedana.gscontrol as gsc
from tedana import io
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_denoising(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Generate and save denoised data.

    This stage writes the denoised time series by removing rejected
    components, optionally applies minimum image regression (MIR)
    for global signal control, and optionally writes per-echo
    denoised data in verbose mode.

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
    # Write denoised results
    io.writeresults(
        state.data_optcom,
        mask=state.mask_denoise,
        component_table=state.component_table,
        mixing=state.mixing,
        io_generator=state.io_generator,
    )

    # Apply minimum image regression if requested
    if "mir" in config.gscontrol:
        gsc.minimum_image_regression(
            data_optcom=state.data_optcom,
            mixing=state.mixing,
            mask=state.mask_denoise,
            component_table=state.component_table,
            classification_tags=state.selector.classification_tags,
            io_generator=state.io_generator,
        )

    # Write per-echo results if verbose
    if config.verbose:
        io.writeresults_echoes(
            state.data_cat,
            state.mixing,
            state.mask_denoise,
            state.component_table,
            state.io_generator,
        )

    return StageResult.ok()
