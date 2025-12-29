"""Workflow stages for the tedana pipeline.

This module provides modular stage functions for the tedana workflow,
enabling independent testing and customizable pipelines.

Example
-------
>>> from tedana.workflows.stages import (
...     WorkflowConfig,
...     WorkflowState,
...     run_initialization,
...     run_data_loading,
... )
>>> config = WorkflowConfig(data=["data.nii"], tes=[14.0, 29.0, 44.0], out_dir="output")
>>> state = WorkflowState()
>>> run_initialization(config, state)
>>> run_data_loading(config, state)
"""

from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)
from tedana.workflows.stages.component_processing import run_component_processing
from tedana.workflows.stages.data_loading import run_data_loading
from tedana.workflows.stages.denoising import run_denoising
from tedana.workflows.stages.echo_combination import run_echo_combination
from tedana.workflows.stages.ica_decomposition import run_ica_decomposition
from tedana.workflows.stages.initialization import run_initialization
from tedana.workflows.stages.metrics_classification import (
    run_metrics_and_classification,
)
from tedana.workflows.stages.pca_decomposition import run_pca_decomposition
from tedana.workflows.stages.reporting import run_reporting
from tedana.workflows.stages.t2s_estimation import run_t2s_estimation

__all__ = [
    # Dataclasses
    "WorkflowConfig",
    "WorkflowState",
    "StageResult",
    # Stage functions
    "run_initialization",
    "run_data_loading",
    "run_t2s_estimation",
    "run_echo_combination",
    "run_pca_decomposition",
    "run_ica_decomposition",
    "run_metrics_and_classification",
    "run_component_processing",
    "run_denoising",
    "run_reporting",
]
