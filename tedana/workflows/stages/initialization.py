"""Stage 1: Initialization and setup for the tedana workflow.

This stage handles:
- Output directory creation
- Logging setup
- Echo time validation
- ComponentSelector initialization
- External regressor validation
- OutputGenerator creation
- System info recording
"""

import datetime
import logging
import os
import os.path as op
from glob import glob

from tedana import io, metrics, utils
from tedana.selection.component_selector import ComponentSelector
from tedana.workflows.parser_utils import check_tedpca_value
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_initialization(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Initialize the tedana workflow.

    This stage creates the output directory, sets up logging, validates
    echo times, initializes the ComponentSelector and OutputGenerator,
    and records system information.

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
    warnings_list = []

    # Create output directory
    out_dir = op.abspath(config.out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Setup report file paths
    prefix = io._infer_prefix(config.prefix)
    basename = f"{prefix}report"
    extension = "txt"
    repname = op.join(out_dir, f"{basename}.{extension}")
    bibtex_file = op.join(out_dir, f"{prefix}references.bib")

    # Rename previous reports
    repex = op.join(out_dir, f"{basename}*")
    previousreps = glob(repex)
    previousreps.sort(reverse=True)
    for f in previousreps:
        previousparts = op.splitext(f)
        newname = previousparts[0] + "_old" + previousparts[1]
        os.rename(f, newname)

    # Setup logging
    basename = "tedana_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, f"{basename}{start_time}.{extension}")
    utils.setup_loggers(logname, repname, quiet=config.quiet, debug=config.debug)

    # Save command to shell file
    if config.tedana_command is not None:
        command_file = open(os.path.join(out_dir, "tedana_call.sh"), "w")
        command_file.write(config.tedana_command)
        command_file.close()

    # Build command string for reproducibility
    tedana_command = (
        config.tedana_command if config.tedana_command else _build_command_string(config)
    )

    LGR.info(f"Using output directory: {out_dir}")

    # Validate and normalize echo times
    tes = [float(te) for te in config.tes]
    tes = utils.check_te_values(tes)
    n_echos = len(tes)

    # Validate tedpca value (raises exception if invalid)
    check_tedpca_value(config.tedpca, is_parser=False)

    LGR.info("Initializing and validating component selection tree")
    selector = ComponentSelector(config.tree, out_dir)

    # Load and validate data to get reference image and dimensions
    LGR.info(f"Loading input data: {[f for f in config.data]}")
    data_cat, ref_img = io.load_data(config.data, n_echos=n_echos, dummy_scans=config.dummy_scans)

    # Load external regressors if provided
    external_regressors = None
    if (
        "external_regressor_config" in set(selector.tree.keys())
        and selector.tree["external_regressor_config"] is not None
    ):
        external_regressors, selector.tree["external_regressor_config"] = (
            metrics.external.load_validate_external_regressors(
                external_regressors=config.external_regressors,
                external_regressor_config=selector.tree["external_regressor_config"],
                n_vols=data_cat.shape[2],
                dummy_scans=config.dummy_scans,
            )
        )

    # Create OutputGenerator
    io_generator = io.OutputGenerator(
        ref_img,
        convention=config.convention,
        out_dir=out_dir,
        prefix=config.prefix,
        config="auto",
        overwrite=config.overwrite,
        verbose=config.verbose,
    )

    # Register input files
    io_generator.register_input(config.data)

    # Save system info
    info_dict = utils.get_system_version_info()
    info_dict["Command"] = tedana_command

    # Get data dimensions
    n_samp, n_echos_data, n_vols = data_cat.shape
    LGR.debug(f"Resulting data shape: {data_cat.shape}")

    # Store results in state
    state.data_cat = data_cat
    state.ref_img = ref_img
    state.selector = selector
    state.external_regressors = external_regressors
    state.io_generator = io_generator
    state.info_dict = info_dict
    state.n_echos = n_echos
    state.n_vols = n_vols
    state.n_samp = n_samp
    state.tes = tes
    state.repname = repname
    state.bibtex_file = bibtex_file

    return StageResult(success=True, warnings=warnings_list)


def _build_command_string(config: WorkflowConfig) -> str:
    """Build a reproducible command string from config.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration.

    Returns
    -------
    str
        Command string for reproducibility.
    """
    parts = ["tedana_workflow("]
    parts.append(f"data={config.data}")
    parts.append(f", tes={config.tes}")
    parts.append(f", out_dir='{config.out_dir}'")

    if config.mask:
        parts.append(f", mask='{config.mask}'")
    parts.append(f", convention='{config.convention}'")
    parts.append(f", prefix='{config.prefix}'")
    parts.append(f", dummy_scans={config.dummy_scans}")
    parts.append(f", masktype={config.masktype}")
    parts.append(f", fittype='{config.fittype}'")
    parts.append(f", combmode='{config.combmode}'")
    if config.n_independent_echos:
        parts.append(f", n_independent_echos={config.n_independent_echos}")
    parts.append(f", tree='{config.tree}'")
    if config.external_regressors:
        parts.append(f", external_regressors='{config.external_regressors}'")
    parts.append(f", ica_method='{config.ica_method}'")
    parts.append(f", n_robust_runs={config.n_robust_runs}")
    parts.append(f", tedpca='{config.tedpca}'")
    parts.append(f", fixed_seed={config.fixed_seed}")
    parts.append(f", maxit={config.maxit}")
    parts.append(f", maxrestart={config.maxrestart}")
    parts.append(f", tedort={config.tedort}")
    parts.append(f", gscontrol={config.gscontrol}")
    parts.append(f", no_reports={config.no_reports}")
    parts.append(f", png_cmap='{config.png_cmap}'")
    parts.append(f", verbose={config.verbose}")
    parts.append(f", low_mem={config.low_mem}")
    parts.append(f", debug={config.debug}")
    parts.append(f", quiet={config.quiet}")
    parts.append(f", overwrite={config.overwrite}")
    if config.t2smap:
        parts.append(f", t2smap='{config.t2smap}'")
    if config.mixing_file:
        parts.append(f", mixing_file='{config.mixing_file}'")
    parts.append(f", n_threads={config.n_threads}")
    parts.append(")")

    return "".join(parts)
