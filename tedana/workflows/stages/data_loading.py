"""Stage 2: Data loading and mask creation for the tedana workflow.

This stage handles:
- TR validation
- Mixing file handling (if pre-computed)
- T2* map handling (if pre-computed)
- Brain mask creation or loading
- Adaptive mask creation for denoising and classification
"""

import logging
import os.path as op
import shutil

import numpy as np
from nilearn.masking import compute_epi_mask

from tedana import io, utils
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_data_loading(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Load data and create masks.

    This stage validates the TR, handles pre-provided mixing matrices
    and T2* maps, creates or loads the brain mask, and generates
    adaptive masks for denoising and classification.

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

    Raises
    ------
    OSError
        If TR is 0 or if provided files don't exist.
    """
    warnings_list = []

    # Validate TR
    img_t_r = state.io_generator.reference_img.header.get_zooms()[-1]
    if img_t_r == 0:
        return StageResult.fail(
            "Dataset has a TR of 0. This indicates incorrect header information. "
            "To correct this, we recommend using this snippet:\n"
            "https://gist.github.com/jbteves/032c87aeb080dd8de8861cb151bff5d6\n"
            "to correct your TR to the value it should be."
        )
    state.img_t_r = img_t_r

    # Handle pre-provided mixing file
    mixing_file = config.mixing_file
    if mixing_file is not None and op.isfile(mixing_file):
        mixing_file = op.abspath(mixing_file)
        # Allow users to re-run on same folder
        mixing_name_output = state.io_generator.get_name("ICA mixing tsv")
        mixing_file_new_path = op.join(state.io_generator.out_dir, op.basename(mixing_file))
        if op.basename(mixing_file) != op.basename(mixing_name_output) and not op.isfile(
            mixing_file_new_path
        ):
            shutil.copyfile(mixing_file, mixing_file_new_path)
        else:
            # Add "user_provided" to the mixing file's name if it's identical to the new file name
            # or if there's already a file in the output directory with the same name
            shutil.copyfile(
                mixing_file,
                op.join(state.io_generator.out_dir, f"user_provided_{op.basename(mixing_file)}"),
            )
    elif mixing_file is not None:
        return StageResult.fail("Argument '--mix' must be an existing file.")

    # Handle pre-provided T2* map
    t2smap = config.t2smap
    if t2smap is not None and op.isfile(t2smap):
        t2smap_file = state.io_generator.get_name("t2star img")
        t2smap = op.abspath(t2smap)
        # Allow users to re-run on same folder
        if t2smap != t2smap_file:
            shutil.copyfile(t2smap, t2smap_file)
    elif t2smap is not None:
        return StageResult.fail("Argument 't2smap' must be an existing file.")

    RepLGR.info(
        "TE-dependence analysis was performed on input data using the tedana workflow "
        "\\citep{dupre2021te}."
    )

    # Create or load brain mask
    mask = _create_mask(config, state, t2smap)
    state.mask = mask

    # Create adaptive mask for denoising (threshold=1 good echo)
    mask_denoise, masksum_denoise = utils.make_adaptive_mask(
        state.data_cat,
        mask=mask,
        n_independent_echos=config.n_independent_echos,
        threshold=1,
        methods=config.masktype,
    )
    LGR.debug(f"Retaining {mask_denoise.sum()}/{state.n_samp} samples for denoising")
    state.io_generator.save_file(masksum_denoise, "adaptive mask img")

    # Create adaptive mask for classification (threshold=3 good echoes)
    masksum_clf = masksum_denoise.copy()
    masksum_clf[masksum_clf < 3] = 0
    mask_clf = masksum_clf.astype(bool)

    RepLGR.info(
        "A two-stage masking procedure was applied, in which a liberal mask "
        "(including voxels with good data in at least the first echo) was used for "
        "optimal combination, T2*/S0 estimation, and denoising, while a more conservative mask "
        "(restricted to voxels with good data in at least the first three echoes) was used for "
        "the component classification procedure."
    )
    LGR.debug(f"Retaining {mask_clf.sum()}/{state.n_samp} samples for classification")

    # Store masks in state
    state.mask_denoise = mask_denoise
    state.masksum_denoise = masksum_denoise
    state.mask_clf = mask_clf
    state.masksum_clf = masksum_clf

    return StageResult(success=True, warnings=warnings_list)


def _create_mask(config: WorkflowConfig, state: WorkflowState, t2smap: str) -> "np.ndarray":
    """Create or load brain mask.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration.
    state : WorkflowState
        Workflow state with data.
    t2smap : str or None
        Path to pre-computed T2* map.

    Returns
    -------
    np.ndarray
        Brain mask array.
    """
    mask_file = config.mask

    if mask_file and not t2smap:
        # User-provided mask only
        LGR.info("Using user-defined mask")
        RepLGR.info("A user-defined mask was applied to the data.")
        mask = utils.reshape_niimg(mask_file).astype(int)

    elif t2smap and not mask_file:
        # T2* map only - derive mask from it
        LGR.info("Assuming user-defined T2* map is masked and using it to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = (t2s_limited != 0).astype(int)
        # Store T2* values in state
        state.t2s_limited = t2s_limited
        state.t2s_full = t2s_full

    elif t2smap and mask_file:
        # Both T2* map and mask provided
        LGR.info("Combining user-defined mask and T2* map to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = utils.reshape_niimg(mask_file).astype(int)
        mask[t2s_limited == 0] = 0  # reduce mask based on T2* map
        # Store T2* values in state
        state.t2s_limited = t2s_limited
        state.t2s_full = t2s_full

    else:
        # No mask or T2* map - compute EPI mask
        LGR.warning(
            "Computing EPI mask from first echo using nilearn's compute_epi_mask function. "
            "Most external pipelines include more reliable masking functions. "
            "It is strongly recommended to provide an external mask, "
            "and to visually confirm that mask accurately conforms to data boundaries."
        )
        first_echo_img = io.new_nii_like(state.io_generator.reference_img, state.data_cat[:, 0, :])
        mask = compute_epi_mask(first_echo_img).get_fdata()
        mask = utils.reshape_niimg(mask).astype(int)
        RepLGR.info(
            "An initial mask was generated from the first echo using "
            "nilearn's compute_epi_mask function."
        )

    return mask
