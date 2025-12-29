"""Stage 5: PCA decomposition for the tedana workflow.

This stage handles:
- PCA-based dimensionality reduction (tedpca)
- Saving PCA results and whitened data
"""

import logging

from tedana import decomposition, utils
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_pca_decomposition(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Perform PCA dimensionality reduction.

    This stage runs tedpca to reduce the dimensionality of the data
    and estimate the number of components for ICA.

    Can be skipped if a mixing matrix is provided.

    Parameters
    ----------
    config : WorkflowConfig
        Immutable workflow configuration.
    state : WorkflowState
        Mutable workflow state (modified in place).

    Returns
    -------
    StageResult
        Execution result; skipped=True if mixing_file was provided.
    """
    # Skip if mixing matrix was provided
    if config.mixing_file is not None:
        return StageResult.skip("Pre-computed mixing matrix was provided")

    # Identify and remove thermal noise from data
    data_reduced, n_components = decomposition.tedpca(
        state.data_cat,
        state.data_optcom,
        state.mask_clf,
        state.masksum_clf,
        state.io_generator,
        tes=state.tes,
        n_independent_echos=config.n_independent_echos,
        algorithm=config.tedpca,
        kdaw=10.0,
        rdaw=1.0,
        low_mem=config.low_mem,
    )

    # Save whitened image if verbose
    if config.verbose:
        state.io_generator.save_file(
            utils.unmask(data_reduced, state.mask_clf),
            "whitened img",
        )

    # Store results in state
    state.data_reduced = data_reduced
    state.n_components = n_components

    return StageResult.ok()
