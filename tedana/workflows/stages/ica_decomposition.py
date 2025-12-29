"""Stage 6: ICA decomposition for the tedana workflow.

This stage handles:
- ICA decomposition (FastICA or RobustICA)
- Restart loop for convergence failures
- Handling when no BOLD components are found
"""

import logging

import pandas as pd

from tedana import decomposition, metrics, selection
from tedana.selection.component_selector import ComponentSelector
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_ica_decomposition(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Perform ICA decomposition with robustness handling.

    This stage runs ICA (FastICA or RobustICA) with a restart loop
    for handling convergence failures and cases where no BOLD
    components are found.

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
    warnings_list = []

    # If mixing matrix was provided, load it and run metrics/selection
    if config.mixing_file is not None:
        return _run_with_provided_mixing(config, state, warnings_list)

    # Run ICA with restart loop
    return _run_ica_with_restarts(config, state, warnings_list)


def _run_with_provided_mixing(
    config: WorkflowConfig,
    state: WorkflowState,
    warnings_list: list,
) -> StageResult:
    """Run metrics and selection with a user-provided mixing matrix.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration.
    state : WorkflowState
        Workflow state.
    warnings_list : list
        List to append warnings to.

    Returns
    -------
    StageResult
        Execution result.
    """
    LGR.info("Using supplied mixing matrix from ICA")
    mixing = pd.read_table(config.mixing_file).values

    # Get necessary metrics
    necessary_metrics = state.selector.necessary_metrics
    # The figures require some metrics that might not be used by the decision tree
    extra_metrics = ["variance explained", "normalized variance explained", "kappa", "rho"]
    necessary_metrics = sorted(list(set(necessary_metrics + extra_metrics)))

    # Generate metrics
    component_table, mixing = metrics.collect.generate_metrics(
        data_cat=state.data_cat,
        data_optcom=state.data_optcom,
        mixing=mixing,
        adaptive_mask=state.masksum_clf,
        tes=state.tes,
        n_independent_echos=config.n_independent_echos,
        io_generator=state.io_generator,
        label="ICA",
        metrics=necessary_metrics,
        external_regressors=state.external_regressors,
        external_regressor_config=state.selector.tree["external_regressor_config"],
    )

    # Run automatic selection
    selector = selection.automatic_selection(
        component_table,
        state.selector,
        n_echos=state.n_echos,
        n_vols=state.n_vols,
        n_independent_echos=config.n_independent_echos,
    )

    if selector.n_likely_bold_comps_ == 0:
        warning_msg = "No BOLD components found with user-provided ICA mixing matrix."
        LGR.warning(warning_msg)
        warnings_list.append(warning_msg)

    # Store results in state
    state.mixing = mixing
    state.component_table = component_table
    state.selector = selector

    return StageResult(
        success=True,
        skipped=True,
        skip_reason="Using provided mixing matrix",
        warnings=warnings_list,
    )


def _run_ica_with_restarts(
    config: WorkflowConfig,
    state: WorkflowState,
    warnings_list: list,
) -> StageResult:
    """Run ICA decomposition with restart loop for robustness.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration.
    state : WorkflowState
        Workflow state.
    warnings_list : list
        List to append warnings to.

    Returns
    -------
    StageResult
        Execution result.
    """
    keep_restarting = True
    n_restarts = 0
    seed = config.fixed_seed

    while keep_restarting:
        # Run ICA
        (
            mixing,
            seed,
            cluster_labels,
            similarity_t_sne,
            fastica_convergence_warning_count,
            index_quality,
        ) = decomposition.tedica(
            state.data_reduced,
            state.n_components,
            seed,
            config.ica_method,
            config.n_robust_runs,
            config.maxit,
            maxrestart=(config.maxrestart - n_restarts),
            n_threads=config.n_threads,
        )
        seed += 1
        n_restarts = seed - config.fixed_seed

        # Get necessary metrics
        necessary_metrics = state.selector.necessary_metrics
        # The figures require some metrics that might not be used by the decision tree
        extra_metrics = ["variance explained", "normalized variance explained", "kappa", "rho"]
        necessary_metrics = sorted(list(set(necessary_metrics + extra_metrics)))

        # Generate metrics
        component_table, mixing = metrics.collect.generate_metrics(
            data_cat=state.data_cat,
            data_optcom=state.data_optcom,
            mixing=mixing,
            adaptive_mask=state.masksum_clf,
            tes=state.tes,
            n_independent_echos=config.n_independent_echos,
            io_generator=state.io_generator,
            label="ICA",
            metrics=necessary_metrics,
            external_regressors=state.external_regressors,
            external_regressor_config=state.selector.tree["external_regressor_config"],
        )

        LGR.info("Selecting components from ICA results")
        selector = selection.automatic_selection(
            component_table,
            state.selector,
            n_echos=state.n_echos,
            n_vols=state.n_vols,
            n_independent_echos=config.n_independent_echos,
        )
        n_likely_bold_comps = selector.n_likely_bold_comps_

        if n_likely_bold_comps == 0:
            if config.ica_method.lower() == "robustica":
                warning_msg = "No BOLD components found with robustICA mixing matrix."
                LGR.warning(warning_msg)
                warnings_list.append(warning_msg)
                keep_restarting = False
            elif n_restarts >= config.maxrestart:
                warning_msg = "No BOLD components found, but maximum number of restarts reached."
                LGR.warning(warning_msg)
                warnings_list.append(warning_msg)
                keep_restarting = False
            else:
                LGR.warning("No BOLD components found. Re-attempting ICA.")
                # If we're going to restart, temporarily allow force overwrite
                state.io_generator.overwrite = True
                # Create a re-initialized selector object if rerunning
                # Since external_regressor_config might have been expanded to remove
                # regular expressions immediately after initialization,
                # store and copy this key
                tmp_external_regressor_config = selector.tree["external_regressor_config"]
                selector = ComponentSelector(config.tree)
                selector.tree["external_regressor_config"] = tmp_external_regressor_config
                RepLGR.disabled = True  # Disable the report to avoid duplicate text
                state.selector = selector
        else:
            keep_restarting = False

    RepLGR.disabled = False  # Re-enable the report after the while loop is escaped
    state.io_generator.overwrite = config.overwrite  # Re-enable original overwrite behavior

    # Store RobustICA-specific metrics if applicable
    if config.ica_method.lower() == "robustica":
        selector.cross_component_metrics_["fastica_convergence_warning_count"] = (
            fastica_convergence_warning_count
        )
        selector.cross_component_metrics_["robustica_mean_index_quality"] = index_quality

    # Store results in state
    state.mixing = mixing
    state.component_table = component_table
    state.selector = selector
    state.cluster_labels = cluster_labels
    state.similarity_t_sne = similarity_t_sne
    state.fastica_convergence_warning_count = fastica_convergence_warning_count

    return StageResult(success=True, warnings=warnings_list)
