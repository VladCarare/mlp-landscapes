# Run the salicylic acid / ANI2x example end to end:
#
#     source RUNME.sh
#
# Both topsearch submodules install as the distribution `topsearch`, so only
# one of them is importable at a time. This script installs the mlp_run branch
# to generate the landscapes, then swaps to the analysis branch for the
# analysis stages, checking after each swap that the intended branch is the
# one in use. Every stage is checked, so a failure stops the run at the cause
# rather than cascading into confusing errors further down.

mlp_landscapes_runme() {

    local example='examples/salicylic_acid_ani2x'
    local runs="${example}/landscape_runs"

    if [ ! -f external/topsearch-mlp_run/pyproject.toml ] ||
       [ ! -f external/topsearch-analysis/pyproject.toml ]; then
        echo 'ERROR: the topsearch submodules are missing. Clone with' >&2
        echo '  git clone --recursive https://github.com/VladCarare/mlp-landscapes.git' >&2
        echo 'or, in an existing clone, run' >&2
        echo '  git submodule update --init --recursive' >&2
        return 1
    fi

    # Re-running the script should reuse the environment rather than fail.
    if conda env list | grep -q '^mlp_landscapes[[:space:]]'; then
        echo 'Conda environment mlp_landscapes already exists, reusing it.'
    else
        conda create -y -n mlp_landscapes python=3.11 || return 1
    fi
    conda activate mlp_landscapes || return 1

    # torchani 2.2.4 imports pkg_resources, which setuptools removed in v81.
    pip install 'setuptools<81' || return 1
    pip install torch==2.4.0 || return 1
    pip install torchani==2.2.4 || return 1

    echo 'Installing topsearch branch that was used to generate the KTNs.'
    pip install -e external/topsearch-mlp_run/ || return 1
    python scripts/check_topsearch_branch.py mlp_run || return 1

    echo "Running salicylic acid small example with ANI2x. Allow a few minutes."
    python "${example}/run_landscape_runs.py" || return 1

    echo 'Finished. Converting files to .xyz. Find them in the corresponding folders:'
    local seed
    for seed in 0 1 2; do
        (
            cd "${runs}/seed${seed}" &&
            python ../../../../scripts/convert_min_coords_to_atoms.py ../../data/salicylic_acid_3_structures.xyz &&
            python ../../../../scripts/convert_ts_coords_to_atoms.py ../../data/salicylic_acid_3_structures.xyz
        ) || return 1
        echo "${runs}/seed${seed}"
    done

    echo 'Installing topsearch branch that was used for analysis.'
    pip install -e external/topsearch-analysis/ || return 1
    python scripts/check_topsearch_branch.py analysis || return 1

    python "${example}/combine_results.py" || return 1

    (
        cd "${runs}" &&
        python ../../../scripts/convert_min_coords_to_atoms.py ../data/salicylic_acid_ground_state_canon_perm.xyz &&
        python ../../../scripts/convert_ts_coords_to_atoms.py ../data/salicylic_acid_ground_state_canon_perm.xyz
    ) || return 1

    echo 'Counting non-physical stationary points. Saving results to file.'
    python "${example}/count_unphysical_stationary_points.py" \
        > "${runs}/output_count_unphysical_stationary_points.out" || return 1

    echo 'Running exact matches comparison. Saving results to file.'
    python "${example}/analyse_exact_matches.py" \
        > "${runs}/output_analysis_exact_matches.out" || return 1

    echo 'Running closest matches comparison. Saving results to file.'
    python "${example}/analyse_closest_matches.py" \
        > "${runs}/output_analysis_closest_matches.out" || return 1

    echo 'Installing PyGT for rates.'
    pip install PyGT || return 1
    pip install pandas || return 1

    echo 'Computing rates. Saving results to file.'
    python "${example}/compute_rates_and_plot.py" \
        > "${runs}/output_compute_rates.out" || return 1

    echo 'Plotting.'
    python "${example}/plotter.py" || return 1

    echo
    echo "Done. Results and plots are in ${runs}/"
}

mlp_landscapes_runme
mlp_landscapes_runme_status=$?
unset -f mlp_landscapes_runme

if [ "${mlp_landscapes_runme_status}" -ne 0 ]; then
    echo 'RUNME.sh stopped at the failing step above.' >&2
fi
