"""Stage 10: Reporting for the tedana workflow.

This stage handles:
- Saving registry of outputs
- Writing BIDS-compatible description file
- Finalizing report text and BibTeX references
- Generating static figures
- Generating dynamic HTML report
- Logging completion and cleanup
"""

import json
import logging
import os

from tedana import __version__, io, reporting, utils
from tedana.bibtex import get_description_references
from tedana.workflows.stages._dataclasses import (
    StageResult,
    WorkflowConfig,
    WorkflowState,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def run_reporting(
    config: WorkflowConfig,
    state: WorkflowState,
) -> StageResult:
    """Generate workflow outputs and reports.

    This stage saves the registry of outputs, writes BIDS-compatible
    description files, finalizes the report text and BibTeX references,
    generates static figures (if reports enabled), and generates
    the dynamic HTML report.

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
    # Write out registry of outputs
    state.io_generator.save_self()

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "tedana Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "tedana",
                "Version": __version__,
                "Description": (
                    "A denoising pipeline for the identification and removal "
                    "of non-BOLD noise from multi-echo fMRI data."
                ),
                "CodeURL": "https://github.com/ME-ICA/tedana",
                "Node": {
                    "Name": state.info_dict["Node"],
                    "System": state.info_dict["System"],
                    "Machine": state.info_dict["Machine"],
                    "Processor": state.info_dict["Processor"],
                    "Release": state.info_dict["Release"],
                    "Version": state.info_dict["Version"],
                },
                "Python": state.info_dict["Python"],
                "Python_Libraries": state.info_dict["Python_Libraries"],
                "Command": state.info_dict["Command"],
            }
        ],
    }
    with open(state.io_generator.get_name("data description json"), "w") as fo:
        json.dump(derivative_metadata, fo, sort_keys=True, indent=4)

    # Add library citations to report
    RepLGR.info(
        "\n\nThis workflow used numpy \\citep{van2011numpy}, scipy \\citep{virtanen2020scipy}, "
        "pandas \\citep{mckinney2010data,reback2020pandas}, "
        "scikit-learn \\citep{pedregosa2011scikit}, "
        "nilearn, bokeh \\citep{bokehmanual}, matplotlib \\citep{Hunter2007}, "
        "and nibabel \\citep{brett_matthew_2019_3233118}."
    )

    RepLGR.info(
        "This workflow also used the Dice similarity index "
        "\\citep{dice1945measures,sorensen1948method}."
    )

    # Finalize report text
    with open(state.repname) as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = " ".join(report)
        # Double-spaces reflect new paragraphs
        report = report.replace("  ", "\n\n")

    with open(state.repname, "w") as fo:
        fo.write(report)

    # Collect BibTeX entries for cited papers
    references = get_description_references(report)

    with open(state.bibtex_file, "w") as fo:
        fo.write(references)

    # Generate figures if reports are enabled
    if not config.no_reports:
        _generate_figures(config, state)

    LGR.info("Workflow completed")

    # Add newsletter info to the log
    utils.log_newsletter_info()

    utils.teardown_loggers()

    return StageResult.ok()


def _generate_figures(config: WorkflowConfig, state: WorkflowState) -> None:
    """Generate static and dynamic figures for the report.

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration.
    state : WorkflowState
        Workflow state.
    """
    import pandas as pd

    LGR.info("Making figures folder with static component maps and timecourse plots.")

    # Get denoised data for carpet plots
    data_denoised, data_accepted, data_rejected = io.denoise_ts(
        state.data_optcom,
        state.mixing,
        state.mask_denoise,
        state.component_table,
    )

    # Generate adaptive mask plot
    reporting.static_figures.plot_adaptive_mask(
        optcom=state.data_optcom,
        base_mask=state.mask,
        io_generator=state.io_generator,
    )

    # Generate carpet plots
    reporting.static_figures.carpet_plot(
        optcom_ts=state.data_optcom,
        denoised_ts=data_denoised,
        hikts=data_accepted,
        lowkts=data_rejected,
        mask=state.mask_denoise,
        io_generator=state.io_generator,
        gscontrol=config.gscontrol,
    )

    # Generate component figures
    reporting.static_figures.comp_figures(
        state.data_optcom,
        mask=state.mask_denoise,
        component_table=state.component_table,
        mixing=state.mixing_orig,
        io_generator=state.io_generator,
        png_cmap=config.png_cmap,
    )

    # Generate T2* and S0 plots
    reporting.static_figures.plot_t2star_and_s0(
        io_generator=state.io_generator,
        mask=state.mask_denoise,
    )

    # Generate RMSE plot if T2* was computed (not provided)
    if config.t2smap is None:
        reporting.static_figures.plot_rmse(
            io_generator=state.io_generator,
            adaptive_mask=state.masksum_denoise,
        )

    # Generate global signal control plots if used
    if config.gscontrol:
        reporting.static_figures.plot_gscontrol(
            io_generator=state.io_generator,
            gscontrol=config.gscontrol,
        )

    # Generate external regressor heatmap if used
    if state.external_regressors is not None:
        comp_names = state.component_table["Component"].values
        mixing_df = pd.DataFrame(data=state.mixing, columns=comp_names)
        reporting.static_figures.plot_heatmap(
            mixing=mixing_df,
            external_regressors=state.external_regressors,
            component_table=state.component_table,
            out_file=os.path.join(
                state.io_generator.out_dir,
                "figures",
                f"{state.io_generator.prefix}confound_correlations.svg",
            ),
        )

    LGR.info("Generating dynamic report")
    reporting.generate_report(
        state.io_generator,
        state.cluster_labels,
        state.similarity_t_sne,
    )
