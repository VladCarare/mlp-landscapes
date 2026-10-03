#!/usr/bin/env bash
# Reproduce the salicylic acid landscape with AIMNet2.
#
#     bash examples/salicylic_acid_aimnet2/reproduce.sh
#
# Run from the repository root. Uses the same scripts as the ANI2x example;
# only the environment below differs. The reference structures and the DFT
# network are properties of the molecule rather than the model, so they are
# read from the ANI2x example's data directory rather than duplicated.
#
# Seeds are independent, so they are split across parallel processes. Set
# JOBS to control how many run at once, and SEEDS to change the set.
set -uo pipefail

export MLP_LANDSCAPES_EXAMPLE=examples/salicylic_acid_aimnet2
export MLP_LANDSCAPES_DATA=examples/salicylic_acid_ani2x/data
export MLP_LANDSCAPES_MODEL=aimnet2
export MLP_LANDSCAPES_BH_STEPS="${MLP_LANDSCAPES_BH_STEPS:-50}"

SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
JOBS="${JOBS:-8}"
RUNS="${MLP_LANDSCAPES_EXAMPLE}/landscape_runs"
EXAMPLE_SCRIPTS=examples/salicylic_acid_ani2x

mkdir -p "${RUNS}"

echo "model=${MLP_LANDSCAPES_MODEL} steps=${MLP_LANDSCAPES_BH_STEPS} seeds=${SEEDS} jobs=${JOBS}"

# One process per seed, at most JOBS at a time. Each writes its own
# landscape_runs/seed<n>/, so they never touch the same files.
pids=()
for seed in ${SEEDS//,/ }; do
    if [ -f "${RUNS}/seed${seed}/min.data" ]; then
        echo "seed ${seed}: already done, skipping"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do sleep 5; done
    ( MLP_LANDSCAPES_SEEDS="${seed}" \
      python "${EXAMPLE_SCRIPTS}/run_landscape_runs.py" \
          > "${RUNS}/seed${seed}.log" 2>&1 \
      && echo "seed ${seed}: done" \
      || echo "seed ${seed}: FAILED, see ${RUNS}/seed${seed}.log" ) &
    pids+=($!)
done
wait "${pids[@]}" 2>/dev/null

echo
echo "Generation finished. Analysis stages:"
set -x
python "${EXAMPLE_SCRIPTS}/combine_results.py"                      || exit 1
( cd "${RUNS}" && python ../../../scripts/convert_min_coords_to_atoms.py \
      ../../salicylic_acid_ani2x/data/salicylic_acid_ground_state_canon_perm.xyz ) || exit 1
( cd "${RUNS}" && python ../../../scripts/convert_ts_coords_to_atoms.py \
      ../../salicylic_acid_ani2x/data/salicylic_acid_ground_state_canon_perm.xyz ) || exit 1
python "${EXAMPLE_SCRIPTS}/count_unphysical_stationary_points.py" \
    > "${RUNS}/output_count_unphysical_stationary_points.out"       || exit 1
python "${EXAMPLE_SCRIPTS}/analyse_exact_matches.py" \
    > "${RUNS}/output_analysis_exact_matches.out"                   || exit 1
python "${EXAMPLE_SCRIPTS}/analyse_closest_matches.py" \
    > "${RUNS}/output_analysis_closest_matches.out"                 || exit 1
python "${EXAMPLE_SCRIPTS}/compute_rates_and_plot.py" \
    > "${RUNS}/output_compute_rates.out"                            || exit 1
python "${EXAMPLE_SCRIPTS}/plotter.py"                              || exit 1
set +x
echo "Done. Results in ${RUNS}/"
