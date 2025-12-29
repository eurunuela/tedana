"""Dataclasses for the tedana workflow stages.

This module defines the core data structures used to pass configuration
and state between workflow stages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from tedana.config import (
    DEFAULT_ICA_METHOD,
    DEFAULT_N_MAX_ITER,
    DEFAULT_N_MAX_RESTART,
    DEFAULT_N_ROBUST_RUNS,
    DEFAULT_SEED,
)


@dataclass
class WorkflowConfig:
    """Immutable configuration parameters for the tedana workflow.

    This dataclass holds all input parameters that control workflow behavior.
    It should not be modified after initialization.

    Parameters
    ----------
    data : list of str
        List of paths to input multi-echo fMRI data files.
    tes : list of float
        Echo times in milliseconds.
    out_dir : str
        Output directory path.
    mask : str or None
        Path to brain mask file.
    convention : str
        File naming convention ('bids' or 'orig').
    prefix : str
        Prefix for output filenames.
    dummy_scans : int
        Number of dummy scans to remove.
    masktype : list of str
        Methods for adaptive mask creation.
    fittype : str
        T2* fitting method ('loglin' or 'curvefit').
    combmode : str
        Echo combination mode ('t2s').
    n_independent_echos : int or None
        Number of independent echoes for metrics.
    tree : str
        Decision tree name or path.
    external_regressors : str or None
        Path to external regressors file.
    ica_method : str
        ICA method ('fastica' or 'robustica').
    n_robust_runs : int
        Number of robust ICA runs.
    tedpca : str, int, or float
        PCA algorithm or component count.
    fixed_seed : int
        Random seed for reproducibility.
    maxit : int
        Maximum ICA iterations.
    maxrestart : int
        Maximum ICA restarts.
    tedort : bool
        Whether to orthogonalize rejected components.
    gscontrol : list of str or None
        Global signal control methods.
    no_reports : bool
        Whether to skip report generation.
    png_cmap : str
        Colormap for PNG figures.
    verbose : bool
        Whether to enable verbose output.
    low_mem : bool
        Whether to use low memory mode.
    debug : bool
        Whether to enable debug logging.
    quiet : bool
        Whether to suppress output.
    overwrite : bool
        Whether to overwrite existing files.
    t2smap : str or None
        Path to pre-computed T2* map.
    mixing_file : str or None
        Path to pre-computed mixing matrix.
    n_threads : int
        Number of threads for parallel processing.
    tedana_command : str or None
        The command used to invoke tedana.
    """

    # Required inputs
    data: List[str]
    tes: List[float]

    # Output configuration
    out_dir: str = "."
    convention: str = "bids"
    prefix: str = ""
    overwrite: bool = False
    verbose: bool = False

    # Preprocessing options
    mask: Optional[str] = None
    dummy_scans: int = 0
    masktype: List[str] = field(default_factory=lambda: ["dropout"])

    # T2* fitting options
    fittype: str = "loglin"
    combmode: str = "t2s"
    n_independent_echos: Optional[int] = None

    # Decision tree options
    tree: str = "tedana_orig"
    external_regressors: Optional[str] = None

    # ICA options
    ica_method: str = DEFAULT_ICA_METHOD
    n_robust_runs: int = DEFAULT_N_ROBUST_RUNS
    tedpca: Union[str, int, float] = "aic"
    fixed_seed: int = DEFAULT_SEED
    maxit: int = DEFAULT_N_MAX_ITER
    maxrestart: int = DEFAULT_N_MAX_RESTART

    # Post-processing options
    tedort: bool = False
    gscontrol: Optional[List[str]] = None

    # Reporting options
    no_reports: bool = False
    png_cmap: str = "coolwarm"

    # Runtime options
    low_mem: bool = False
    debug: bool = False
    quiet: bool = False
    n_threads: int = 1

    # Pre-computed inputs
    t2smap: Optional[str] = None
    mixing_file: Optional[str] = None

    # Command tracking
    tedana_command: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Ensure data is a list
        if isinstance(self.data, str):
            self.data = [self.data]

        # Ensure gscontrol is a list
        if self.gscontrol is None:
            self.gscontrol = [None]
        elif not isinstance(self.gscontrol, list):
            self.gscontrol = [self.gscontrol]


@dataclass
class WorkflowState:
    """Mutable state that evolves through the workflow stages.

    This dataclass holds all intermediate results and data that are
    computed and passed between workflow stages.

    Attributes
    ----------
    data_cat : np.ndarray or None
        Multi-echo data array with shape (S, E, T) where S is samples,
        E is echoes, and T is time points.
    data_optcom : np.ndarray or None
        Optimally combined data with shape (S, T).
    data_reduced : np.ndarray or None
        PCA-reduced data.
    ref_img : nibabel image or None
        Reference NIfTI image for output.
    mask : np.ndarray or None
        Base brain mask.
    mask_denoise : np.ndarray or None
        Liberal mask for denoising (threshold=1 good echo).
    mask_clf : np.ndarray or None
        Conservative mask for classification (threshold=3 good echoes).
    masksum_denoise : np.ndarray or None
        Count of good echoes per voxel for denoising.
    masksum_clf : np.ndarray or None
        Count of good echoes per voxel for classification.
    t2s_limited : np.ndarray or None
        Limited T2* map.
    s0_limited : np.ndarray or None
        Limited S0 map.
    t2s_full : np.ndarray or None
        Full T2* map.
    s0_full : np.ndarray or None
        Full S0 map.
    n_components : int or None
        Number of ICA components.
    mixing : np.ndarray or None
        ICA mixing matrix.
    mixing_orig : np.ndarray or None
        Original mixing matrix before TEDORT.
    component_table : pd.DataFrame or None
        Component metrics and classifications.
    selector : ComponentSelector or None
        Component selection object.
    cluster_labels : np.ndarray or None
        RobustICA cluster labels.
    similarity_t_sne : np.ndarray or None
        RobustICA t-SNE similarity.
    fastica_convergence_warning_count : int or None
        Count of FastICA convergence warnings.
    external_regressors : pd.DataFrame or None
        Loaded external regressors.
    io_generator : OutputGenerator or None
        Output file manager.
    info_dict : dict or None
        System and command information.
    n_echos : int or None
        Number of echoes.
    n_vols : int or None
        Number of time points.
    n_samp : int or None
        Number of samples (voxels).
    img_t_r : float or None
        Repetition time.
    repname : str or None
        Path to report text file.
    bibtex_file : str or None
        Path to BibTeX file.
    tes : list of float or None
        Validated echo times in milliseconds.
    """

    # Data arrays
    data_cat: Optional[np.ndarray] = None
    data_optcom: Optional[np.ndarray] = None
    data_reduced: Optional[np.ndarray] = None
    ref_img: Optional[Any] = None  # nibabel image

    # Masks
    mask: Optional[np.ndarray] = None
    mask_denoise: Optional[np.ndarray] = None
    mask_clf: Optional[np.ndarray] = None
    masksum_denoise: Optional[np.ndarray] = None
    masksum_clf: Optional[np.ndarray] = None

    # T2*/S0 estimates
    t2s_limited: Optional[np.ndarray] = None
    s0_limited: Optional[np.ndarray] = None
    t2s_full: Optional[np.ndarray] = None
    s0_full: Optional[np.ndarray] = None

    # Decomposition results
    n_components: Optional[int] = None
    mixing: Optional[np.ndarray] = None
    mixing_orig: Optional[np.ndarray] = None

    # Component table and selector
    component_table: Optional[pd.DataFrame] = None
    selector: Optional[Any] = None  # ComponentSelector

    # RobustICA results
    cluster_labels: Optional[np.ndarray] = None
    similarity_t_sne: Optional[np.ndarray] = None
    fastica_convergence_warning_count: Optional[int] = None

    # External regressors
    external_regressors: Optional[pd.DataFrame] = None

    # I/O components
    io_generator: Optional[Any] = None  # OutputGenerator

    # Metadata
    info_dict: Optional[Dict[str, Any]] = None
    n_echos: Optional[int] = None
    n_vols: Optional[int] = None
    n_samp: Optional[int] = None
    img_t_r: Optional[float] = None

    # Report paths
    repname: Optional[str] = None
    bibtex_file: Optional[str] = None

    # Validated echo times
    tes: Optional[List[float]] = None


@dataclass
class StageResult:
    """Result from a workflow stage execution.

    This dataclass provides structured feedback about stage execution,
    supporting success, failure, and skip scenarios.

    Parameters
    ----------
    success : bool
        Whether the stage completed successfully.
    skipped : bool
        Whether the stage was skipped (e.g., due to pre-computed inputs).
    skip_reason : str or None
        Explanation for why the stage was skipped.
    warnings : list of str
        Warning messages generated during execution.
    errors : list of str
        Error messages generated during execution.
    """

    success: bool
    skipped: bool = False
    skip_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "StageResult":
        """Create a successful result."""
        return cls(success=True)

    @classmethod
    def skip(cls, reason: str) -> "StageResult":
        """Create a skipped result with explanation."""
        return cls(success=True, skipped=True, skip_reason=reason)

    @classmethod
    def fail(cls, error: str) -> "StageResult":
        """Create a failed result with error message."""
        return cls(success=False, errors=[error])
