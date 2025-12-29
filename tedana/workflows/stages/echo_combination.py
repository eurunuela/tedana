"""Stage 4: Echo combination for the tedana workflow.

This stage handles:
- Optimal combination of multi-echo data
- Optional global signal regression (GSR)
- Saving the optimally combined dataset
"""

import logging

import tedana.gscontrol as gsc
from tedana import combine
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_echo_combination(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Optimally combine echoes and optionally apply GSR.

    This stage combines echoes using T2*-weighted averaging,
    optionally regresses out the global signal, and saves
    the optimally combined data.

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
    # Optimally combine echoes
    data_optcom = combine.make_optcom(
        state.data_cat,
        state.tes,
        state.masksum_denoise,
        t2s=state.t2s_full,
        combmode=config.combmode,
    )

    # Apply global signal regression if requested
    if "gsr" in config.gscontrol:
        data_cat, data_optcom = gsc.gscontrol_raw(
            data_cat=state.data_cat,
            data_optcom=data_optcom,
            n_echos=state.n_echos,
            io_generator=state.io_generator,
        )
        # Update state.data_cat if GSR was applied
        state.data_cat = data_cat

    # Save optimally combined data
    fout = state.io_generator.save_file(data_optcom, "combined img")
    LGR.info(f"Writing optimally combined data set: {fout}")

    # Store result in state
    state.data_optcom = data_optcom

    # Initialize RobustICA results to None (will be set if used)
    state.cluster_labels = None
    state.similarity_t_sne = None
    state.fastica_convergence_warning_count = None

    return StageResult.ok()
