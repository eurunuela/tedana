"""Stage 7: Metrics calculation and component classification for the tedana workflow.

Note: In the current implementation, metrics calculation and initial
classification happen within the ICA decomposition stage as part of
the restart loop. This stage handles any final metrics updates
and verification.

This stage handles:
- Verification that BOLD components were found
- Any final metrics updates
"""

import logging

from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_metrics_and_classification(
    config: WorkflowConfig,  # noqa: U100
    state: WorkflowState,
) -> StageResult:
    """Verify component classification results.

    This stage verifies that BOLD components were found during
    the ICA decomposition stage and logs appropriate warnings
    if not.

    Note: The main metrics calculation and component selection
    are performed during the ICA decomposition stage to support
    the restart loop logic.

    Parameters
    ----------
    config : WorkflowConfig
        Immutable workflow configuration (unused but kept for API consistency).
    state : WorkflowState
        Mutable workflow state (modified in place).

    Returns
    -------
    StageResult
        Execution result with success/failure status.
    """
    warnings_list = []

    # Check if BOLD components were found
    if state.selector.n_likely_bold_comps_ == 0:
        warning_msg = "No BOLD components detected! Please check data and results!"
        LGR.warning(warning_msg)
        warnings_list.append(warning_msg)

    # Get the final component table from selector
    state.component_table = state.selector.component_table_

    return StageResult(success=True, warnings=warnings_list)
