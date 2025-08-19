conda create -n mlp_landscapes python=3.11

conda activate mlp_landscapes

pip install torch==2.4.0
pip install torchani==2.2.4

echo Installing topsearch branch that was used to generate the KTNs.
pip install -e external/topsearch-mlp_run/

echo Running salicylic acid small example with ANI2x. Allow a few minutes.

python examples/salicylic_acid_ani2x/run_landscape_runs.py

echo Finished. Converting files to .xyz. Find them in the corresponding folders:

for i in 0 1 2; do
    (cd examples/salicylic_acid_ani2x/landscape_runs/seed$i; python ../../../../scripts/convert_min_coords_to_atoms.py ../../data/salicylic_acid_3_structures.xyz; cd -)
    (cd examples/salicylic_acid_ani2x/landscape_runs/seed$i; python ../../../../scripts/convert_ts_coords_to_atoms.py ../../data/salicylic_acid_3_structures.xyz; cd -)
    echo examples/salicylic_acid_ani2x/landscape_runs/seed$i
done

echo Installing topsearch branch that was used for analysis.
pip install -e external/topsearch-analysis/

python examples/salicylic_acid_ani2x/combine_results.py
    
(cd examples/salicylic_acid_ani2x/landscape_runs; python ../../../scripts/convert_min_coords_to_atoms.py ../data/salicylic_acid_ground_state_canon_perm.xyz; cd -)
(cd examples/salicylic_acid_ani2x/landscape_runs; python ../../../scripts/convert_ts_coords_to_atoms.py ../data/salicylic_acid_ground_state_canon_perm.xyz; cd -)


echo Counting non-physical stationary points. Saving results to file.
python examples/salicylic_acid_ani2x/count_unphysical_stationary_points.py > examples/salicylic_acid_ani2x/landscape_runs/output_count_unphysical_stationary_points.out

echo Running exact matches comparison. Saving results to file.
python examples/salicylic_acid_ani2x/analyse_exact_matches.py > examples/salicylic_acid_ani2x/landscape_runs/output_analysis_exact_matches.out

echo Running closest matches comparison. Saving results to file.
python examples/salicylic_acid_ani2x/analyse_closest_matches.py > examples/salicylic_acid_ani2x/landscape_runs/output_analysis_closest_matches.out

echo Installing PyGT for rates.
pip install PyGT
pip install pandas

echo Computing rates. Saving results to file.
python examples/salicylic_acid_ani2x/compute_rates_and_plot.py > examples/salicylic_acid_ani2x/landscape_runs/output_compute_rates.out

echo Plotting.
python examples/salicylic_acid_ani2x/plotter.py