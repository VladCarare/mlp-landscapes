
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle


names = ['Non-Landscape','Landscape','Critical Points\n(Min.&TS)','TS','Min.','TS-L','Min.-L']
colors = ["#f84444", "#2737b2", "#5d93d5","#41b8e3","#91bfd0","#f9b6b6","#ff8c8c"]

color_dict = dict(zip(names,colors))

markers_dict = {
    'Non-Landscape': 'v',
    'Landscape': '^',
    'Critical Points\n(Min.&TS)': 'd',
    'Min.': 'o',
    'TS': 'X'
}
alphas_dict = {
    'Non-Landscape': 0.75,
    'Landscape': 0.75,
    'Critical Points\n(Min.&TS)': 1,
    'Min.': 1,
    'TS': 1,
}
markers_size_dict = {
    'Non-Landscape': 35,
    'Landscape': 35,
    'Critical Points\n(Min.&TS)': 40,
    'Min.': 40,
    'TS': 40
}

model_colors_dict = {'Non-Landscape':{'ani2x':'#a089bd','aimnet2':'#7cd1b8','mace':'#4682b4', 'nequip':'#e07b91', 'allegro':'#a3b600', 'dftb':"#af4949",'mace-mp-0b3':"#004C8E",'so3lr':"#e9da0b"},
                'Landscape': {'mace':'#2e597a', 'nequip':'#a04d64', 'allegro':'#6d7e00'}}

labelfontsize = 10





####### UNPHYSICAL STATIONARY POINTS COUNTS

import pickle 


with open(f"examples/salicylic_acid_ani2x/landscape_runs/count_unphysical_stationary_points.pkl", 'rb') as f:
    nonphysical_min_results = pickle.load(f)

import pandas as pd 

df = pd.DataFrame(nonphysical_min_results)

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches


fig, axs = plt.subplots(1,2,figsize=(6,4),sharex=True)
ax1,ax2 = axs[0],axs[1]
# Set academic style - clean and scientific
plt.rcParams.update({
    # 'font.size': 11,
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'axes.axisbelow': True
})

# Your existing data processing code...
model_order = ['ani2x']

# Process exactly as in your original code
filtered_df = df.iloc[:-1] if len(df) > len(model_order) else df

grouped = (
    filtered_df
    .groupby(["Model", "Type"])[["initial_n_min", "n_physical_min", "initial_n_ts", "n_physical_ts"]]
    .sum()
    .reset_index()
)
grouped['n_nonphysical_min'] = grouped['initial_n_min'] - grouped['n_physical_min']
grouped['n_nonphysical_ts'] = grouped['initial_n_ts'] - grouped['n_physical_ts']
# Reorder models
grouped['Model_order'] = grouped['Model'].apply(lambda x: model_order.index(x) if x in model_order else 999)
grouped = grouped.sort_values('Model_order').drop('Model_order', axis=1)
# Pivot the data
pivot_models = grouped.pivot_table(
    index="Model", 
    columns="Type", 
    values=["n_physical_min", "n_nonphysical_min", "n_physical_ts", "n_nonphysical_ts"]
)
pivot_models = pivot_models.reindex(model_order)

colors = {
    'non_altitude': {
        'min_physical': "#fb6767",     
        'min_nonphysical': "#ff9a9a",    
        'ts_nonphysical': "#ce3a3a",         
        'ts_physical': "#8d2626"    
    },
    'altitude': {
        'min_nonphysical': "#323e9a", 
        'min_physical': "#5761b0",
        'ts_physical': "#24329f",
        'ts_nonphysical': "#131b5b"       
    }
}

# Model labels for display
display_labels = ['ANI2x']

# Plotting function for cleaner code
def plot_stacked_bars(ax, data_type, title, dft_reference, ylabel):

    # Set academic style - clean and scientific
    plt.rcParams.update({
        # 'font.size': 11,
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.dpi': 150,
        'axes.axisbelow': True
    })
    
    x = np.arange(len(model_order))
    bar_width = 0.28

    data_type_label = 'TS' if data_type=='ts' else 'Min.'
    
    # Non-altitude bars (left)
    if (f'n_nonphysical_{data_type}', 'non_altitude') in pivot_models.columns:
        nonphys_non_alt = pivot_models[(f'n_nonphysical_{data_type}', 'non_altitude')].fillna(0)
        phys_non_alt = pivot_models[(f'n_physical_{data_type}', 'non_altitude')].fillna(0)
        
        # Physical (top)
        ax.bar(x - bar_width/2, nonphys_non_alt, bar_width,
            label=f'N-L', 
            color=colors['non_altitude'][f'{data_type}_physical'], 
            alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Altitude bars (right) - only for models that have this data
    if (f'n_nonphysical_{data_type}', 'altitude') in pivot_models.columns:
        for i, model in enumerate(model_order):
            if not pd.isna(pivot_models.loc[model, (f'n_nonphysical_{data_type}', 'altitude')]):
                nonphys_alt = pivot_models.loc[model, (f'n_nonphysical_{data_type}', 'altitude')]
                phys_alt = pivot_models.loc[model, (f'n_physical_{data_type}', 'altitude')]
                
                # Physical (top)  
                ax.bar(i + bar_width/2, nonphys_alt, bar_width,
                    label=f'L' if i == len(model_order)-1 else "",
                    color=colors['altitude'][f'{data_type}_physical'], 
                    alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # DFT reference line
    ax.axhline(dft_reference, linestyle=(0, (10, 8)), color='#2C3E50', linewidth=0.6, 
            alpha=0.6, label=f'# DFT {data_type_label}')
    
    # Formatting
    ax.set_ylabel(ylabel, fontsize=labelfontsize)
    # ax.set_xlabel('ML Models', fontsize=labelfontsize)
    # ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=labelfontsize-1.5,rotation=0)
    
    # Legend with better positioning
    # ax.legend(loc='upper right', fontsize=9, framealpha=0.9, 
    #           fancybox=True, shadow=False, ncol=1)
    # Legend only on bottom
    ax.legend(loc='upper right', fontsize=labelfontsize-1.5, framealpha=0.6, 
                    fancybox=True, shadow=False, ncol=1, bbox_to_anchor=(0.99,1.12))
    
    # Grid styling
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # ax.grid(True, linestyle='--', alpha=0.5,lw=0.8)
    ax.grid(True, linestyle='-', alpha=0.3,lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=labelfontsize-1.5,rotation=35)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    

# Create both subplots
plot_stacked_bars(ax1, 'min', 'Energy Minima Analysis', 28, 'Count [$\longleftarrow$]')
plot_stacked_bars(ax2, 'ts', 'Transition States Analysis', 67, 'Count [$\longleftarrow$]')

print(f"\nDFT Reference: 28 minima, 67 transition states")

plt.savefig('examples/salicylic_acid_ani2x/landscape_runs/count_unphysical_stationary_points.pdf',bbox_inches='tight')





####### UNPHYSICAL STATIONARY POINTS COUNTS


import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

def plot_fig4_perfect_match_comparison_1(fig,n_unphysical_critical_points_plot_gridspec,labelfontsize):

    # molecules = ['ethanol', 'malonaldehyde', 'salicylic','azobenzene','paracetamol','aspirin']
    # models = ['dftb','ani2x','aimnet2','allegro','mace','nequip','mace-mp-0b3','so3lr']
    # types = ['non_altitude','altitude']
    molecules = ['salicylic']
    models = ['ani2x']
    types = ['non_altitude']
    all_results = []

    for molecule in molecules:
        for model in models:
            for model_type in types:
                if (model_type=='altitude' and model in ['aimnet2','ani2x','mace-mp-0b3','dftb','so3lr']):
                    continue
                try:
                    if model_type == 'non_altitude':
                        new_type = 'retrain' # just a naming convention
                    else:
                        new_type = 'altitude'

                    # print(molecule,model,model_type)
                    file = f'examples/salicylic_acid_ani2x/landscape_runs/analysis_exact-matches.pkl'
                    with open(file, 'rb') as f:
                        dft_ktn, ml_ktn, minima_correspondences, missing_ml_minima, statistics = pickle.load(f)
                        n_dft_min = dft_ktn.n_minima
                        n_dft_ts = len(list(dft_ktn.G.edges))
                        d_m = statistics['minima_distances']
                        d_ts = statistics['ts_distances']
                        de_m = statistics['minima_relative_energy_diffs']
                        de_m_absolute = statistics['minima_absolute_energy_diffs']
                        de_ts = statistics['ts_relative_energy_diffs']
                        de_ts_absolute = statistics['ts_absolute_energy_diffs']
                        mm = statistics['missing_minima']
                        mp = statistics['extra_minima']
                        tm = statistics['missing_ts']
                        tp = statistics['extra_ts']

                        d_m = round(np.mean(d_m),5)
                        d_ts = round(np.mean(d_ts),5)
                        de_m = round(np.sqrt(np.mean(np.array(de_m)**2)),4)
                        de_m_absolute = round(np.sqrt(np.mean(np.array(de_m_absolute)**2)),4)
                        de_ts = round(np.sqrt(np.mean(np.array(de_ts)**2)),4)
                        de_ts_absolute = round(np.sqrt(np.mean(np.array(de_ts_absolute)**2)),4)
                        entry={'Molecule':molecule,'Model':model,'Type':model_type,'Avg RMSD min. pairs':d_m,'RMSE E min. pairs (rel. to glob. mins.)':de_m,'RMSE E min. pairs (abs)':de_m_absolute,
                                'Extra mins.':mp,'Missing mins.':mm, 'DFT #min':n_dft_min,
                                'Avg. RMSD TS pairs':d_ts,'RMSE E TS pairs (rel. to glob. mins.)':de_ts,'RMSE E TS pairs (abs)':de_ts_absolute,
                                'Extra TS':tp,'Missing TS':tm, 'DFT #TS':n_dft_ts}
                        all_results.append(entry)

                except FileNotFoundError:
                    print(f'No output file for {model} {molecule} {type}.')
                    pass

    df = pd.DataFrame(all_results)
    df['Matched mins.']=df['DFT #min']-df['Missing mins.']
    df['Matched TS']=df['DFT #TS']-df['Missing TS']

    # Set academic style - clean and scientific
    plt.rcParams.update({
        # 'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.dpi': 150,
        'axes.axisbelow': True
    })

    # Your existing data processing code...
    model_order = ['ani2x']

    grouped = (
        df
        .groupby(["Model", "Type"])[["Missing mins.", "Missing TS"]]
        .sum()
        .reset_index()
    )
    # Reorder models
    grouped['Model_order'] = grouped['Model'].apply(lambda x: model_order.index(x) if x in model_order else 999)
    grouped = grouped.sort_values('Model_order').drop('Model_order', axis=1)
    # Pivot the data
    pivot_models = grouped.pivot_table(
        index="Model", 
        columns="Type", 
        values=["Missing mins.", "Missing TS"]
    )
    pivot_models = pivot_models.reindex(model_order)
    

    # Set academic style - clean and scientific
    plt.rcParams.update({
        # 'font.size': 11,
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.dpi': 150,
        'axes.axisbelow': True
    })

    ax1 = fig.add_subplot(n_unphysical_critical_points_plot_gridspec[0, 0])
    ax2 = fig.add_subplot(n_unphysical_critical_points_plot_gridspec[0, 1])

    colors = {
        'non_altitude': {
            'min_physical': "#fb6767",     
            'min_nonphysical': "#ff9a9a",    
            'ts_nonphysical': "#ce3a3a",         
            'ts_physical': "#8d2626"    
        },
        'altitude': {
            'min_nonphysical': "#323e9a", 
            'min_physical': "#5761b0",
            'ts_physical': "#24329f",
            'ts_nonphysical': "#131b5b"       
        }
    }
    
    # Model labels for display
    display_labels = ['ANI2x']

    # Plotting function for cleaner code
    def plot_stacked_bars(ax, data_type, title, dft_reference, ylabel):

        # Set academic style - clean and scientific
        plt.rcParams.update({
            # 'font.size': 11,
            # 'font.family': 'sans-serif',
            # 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'figure.dpi': 150,
            'axes.axisbelow': True
        })
        
        x = np.arange(len(model_order))
        bar_width = 0.28

        data_type_label = 'TS' if data_type=='TS' else 'Min.'
        data_type_color_label = 'ts' if data_type=='TS' else 'min'
        
        # Non-altitude bars (left)
        if (f'Missing {data_type}', 'non_altitude') in pivot_models.columns:
            missing_non_alt = pivot_models[(f'Missing {data_type}', 'non_altitude')].fillna(0)
            
            # Physical (top)
            ax.bar(x - bar_width/2, missing_non_alt, bar_width,
                label=f'N-L', 
                color=colors['non_altitude'][f'{data_type_color_label}_physical'], 
                alpha=0.8, edgecolor='white', linewidth=0.5)
        
        
        # Altitude bars (right) - only for models that have this data
        if (f'Missing {data_type}', 'altitude') in pivot_models.columns:
            for i, model in enumerate(model_order):
                if not pd.isna(pivot_models.loc[model, (f'Missing {data_type}', 'altitude')]):
                    missing_alt = pivot_models.loc[model, (f'Missing {data_type}', 'altitude')]
                    
                    # Physical (top)  
                    ax.bar(i + bar_width/2, missing_alt, bar_width,
                        label=f'L' if i == len(model_order)-1 else "",
                        color=colors['altitude'][f'{data_type_color_label}_physical'], 
                        alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # # DFT reference line
        # ax.axhline(dft_reference, linestyle=(0, (10, 8)), color='#2C3E50', linewidth=0.6, 
        #         alpha=0.6, label=f'# DFT {data_type_label}')
        
        # Formatting
        ax.set_ylabel(ylabel, fontsize=labelfontsize)
        # ax.set_xlabel('ML Models', fontsize=labelfontsize)
        # ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, fontsize=labelfontsize-1.5,rotation=0)
        
        # Legend with better positioning
        # ax.legend(loc='upper right', fontsize=9, framealpha=0.9, 
        #           fancybox=True, shadow=False, ncol=1)
        # Legend only on bottom
        ax.legend(loc='upper left', fontsize=labelfontsize-1.5, framealpha=0.6, 
                        fancybox=True, shadow=False, ncol=2, bbox_to_anchor=(0.05,1))
        
        # Grid styling
        # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        # ax.grid(True, linestyle='--', alpha=0.5,lw=0.8)
        ax.grid(True, linestyle='-', alpha=0.3,lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=labelfontsize-1.5,rotation=30)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        
        if data_type=='TS':
            ax.set_ylim(0, 70)
        else:
            ax.set_ylim(0,33)

    # Create both subplots
    plot_stacked_bars(ax1, 'mins.', 'Energy Minima Analysis', 28, 'Count [$\longleftarrow$]')
    plot_stacked_bars(ax2, 'TS', 'Transition States Analysis', 67, 'Count [$\longleftarrow$]')


    print(f"\nDFT Reference: 28 minima, 67 transition states")
    return ax1,ax2


import numpy as np
import pickle
import os
from sys import argv
import networkx as nx
import pickle
import numpy as np

import pickle 
import pandas as pd 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches


def plot_fig4_closest_match_RMSD_comparison(fig,closest_match_RMSD_comparison_plot_gridspec,labelfontsize):

    # molecules = ['ethanol', 'malonaldehyde', 'salicylic','azobenzene','paracetamol','aspirin']
    # models = ['ani2x','aimnet2','allegro','mace','nequip','mace-mp-0b3','dftb','so3lr']
    # types = ['non_altitude','altitude']
    molecules = ['salicylic']
    models = ['ani2x']
    types = ['non_altitude']
    all_results = []

    for molecule in molecules:
        for model in models:
            for model_type in types:
                if (model_type=='altitude' and model in ['aimnet2','ani2x','mace-mp-0b3','dftb','so3lr']):
                    continue
                try:
                    if model_type == 'non_altitude':
                        new_type = 'retrain' # just a naming convention
                    else:
                        new_type = 'altitude'
                    
                    file = f'examples/salicylic_acid_ani2x/landscape_runs/analysis_closest-matches.pkl'
                    # if model == 'aimnet2':
                    #     file = f'/Users/vcarare/Downloads/LandscapeWork/landscape-17/landscape_results/{model}_{new_type}_gridsearch/{molecule}/analysis_{model}_{molecule}threshold1e8-14042025landscape-28052025aimnet2Functional-redoneTScomparison.pkl'
                    with open(file, 'rb') as f:
                        dft_ktn, ml_ktn, minima_correspondences, missing_ml_minima, statistics = pickle.load(f)
                        n_dft_min = dft_ktn.n_minima
                        n_dft_ts = len(list(dft_ktn.G.edges))
                        d_m = statistics['minima_distances']
                        d_ts = statistics['ts_distances']
                        de_m = statistics['minima_relative_energy_diffs']
                        de_m_absolute = statistics['minima_absolute_energy_diffs']
                        de_ts = statistics['ts_relative_energy_diffs']
                        de_ts_absolute = statistics['ts_absolute_energy_diffs']
                        mm = statistics['missing_minima']
                        mp = statistics['extra_minima']
                        tm = statistics['missing_ts']
                        tp = statistics['extra_ts']
                        n_atoms = len(dft_ktn.G.nodes[0]['coords'])/3

                        d_m = round(np.mean(d_m),5)
                        d_ts = round(np.mean(d_ts),5)
                        de_m = round(np.sqrt(np.mean((np.array(de_m)/n_atoms)**2)),4)
                        de_m_absolute = round(np.sqrt(np.mean(np.array(de_m_absolute/n_atoms)**2)),4)
                        de_ts = round(np.sqrt(np.mean((np.array(de_ts)/n_atoms)**2)),4)
                        de_ts_absolute = round(np.sqrt(np.mean(np.array(de_ts_absolute/n_atoms)**2)),4)
                        entry={'Molecule':molecule,'Model':model,'Type':model_type,'Avg RMSD min. pairs':d_m,'RMSE E min. pairs (rel. to glob. mins.)':de_m,'RMSE E min. pairs (abs)':de_m_absolute,
                                'Extra mins.':mp,'Missing mins.':mm, 'DFT #min':n_dft_min,
                                'Avg. RMSD TS pairs':d_ts,'RMSE E TS pairs (rel. to glob. mins.)':de_ts,'RMSE E TS pairs (abs)':de_ts_absolute,
                                'Extra TS':tp,'Missing TS':tm, 'DFT #TS':n_dft_ts}
                        all_results.append(entry)

                except FileNotFoundError:
                    print(f'No output file for {model} {molecule} {type}.')
                    pass

    df = pd.DataFrame(all_results)
    
    ax1 = fig.add_subplot(closest_match_RMSD_comparison_plot_gridspec[0,0])
    ax2 = fig.add_subplot(closest_match_RMSD_comparison_plot_gridspec[0,1])

    # bar_width = 0.35  # Width of each bar group
    X = np.arange(len(models))

    # Define the desired order of models
    model_order = ['ani2x']

    # Group by Model and Type, and calculate the mean
    grouped = (
        df
        .groupby(["Model", "Type"])[['Avg RMSD min. pairs', 'Avg. RMSD TS pairs','RMSE E min. pairs (rel. to glob. mins.)','RMSE E TS pairs (rel. to glob. mins.)']]
        .mean()
        .reset_index()
    )

    # Reorder the models according to the specified order
    grouped['Model_order'] = grouped['Model'].apply(lambda x: model_order.index(x) if x in model_order else 999)
    grouped = grouped.sort_values('Model_order').drop('Model_order', axis=1)

    # Pivot the data for plotting
    pivot_models = grouped.pivot_table(
        index="Model", 
        columns="Type", 
        values=['Avg RMSD min. pairs', 'Avg. RMSD TS pairs','RMSE E min. pairs (rel. to glob. mins.)','RMSE E TS pairs (rel. to glob. mins.)']
    )

    # Reindex to ensure the models are in the correct order
    pivot_models = pivot_models.reindex(model_order)

    # For non_altitude type (all models should have this)
    if ('Avg RMSD min. pairs', 'non_altitude') in pivot_models.columns:
        rmsdmins_heights = pivot_models[('Avg RMSD min. pairs', 'non_altitude')]
        for x, rmsd_min in zip(X,rmsdmins_heights):        
            ax1.scatter(x, rmsd_min, marker = markers_dict['Min.'], color= model_colors_dict.get('Non-Landscape').get(model_order[x]), s=80, zorder=5, ec='k',lw=1.)

        rmsemins_heights = pivot_models[('RMSE E min. pairs (rel. to glob. mins.)', 'non_altitude')]
        for x, rmse_min in zip(X[:],rmsemins_heights[:]):        
            ax2.scatter(x, rmse_min*1e3, marker = markers_dict['Min.'], color= model_colors_dict.get('Non-Landscape').get(model_order[x]), s=80, zorder=5, ec='k',lw=1.)


    # For altitude type (some models may not have this)
    if ('Avg RMSD min. pairs', 'altitude') in pivot_models.columns:
        altitude_models = [model for model in pivot_models.index 
                        if model in model_order and not pd.isna(pivot_models.loc[model, ('Avg RMSD min. pairs', 'altitude')])]
        altitude_indices = [model_order.index(model) for model in altitude_models]
        
        rmsdmins_heights = [pivot_models.loc[model, ('Avg RMSD min. pairs', 'altitude')] 
                            for model in altitude_models]
        
        for x, rmsd_min in zip(altitude_indices,rmsdmins_heights):
            ax1.scatter(x, rmsd_min, marker = markers_dict['Min.'], ec= model_colors_dict.get('Landscape').get(model_order[x]), facecolor='w',
                    lw=1.5,s=80, zorder=5)
        
        rmsemins_heights = [pivot_models.loc[model, ('RMSE E min. pairs (rel. to glob. mins.)', 'altitude')] 
                            for model in altitude_models]
        for x, rmse_min in zip(altitude_indices,rmsemins_heights):
            ax2.scatter(x, rmse_min*1e3, marker = markers_dict['Min.'], ec= model_colors_dict.get('Landscape').get(model_order[x]), facecolor='w',
                    lw=1.5,s=80, zorder=5)
        


    ax1.set_ylabel("RMSD ($\AA$) [$\longleftarrow$]", fontsize=labelfontsize)
    ax2.set_ylabel("Rel. E RMSE\n(meV/atom) [$\longleftarrow$]", fontsize=labelfontsize)

    ax1.set_xticklabels(['ANI2x'])
    ax1.set_xticks(X)
    ax2.set_xticklabels(['ANI2x'])
    ax2.set_xticks(X[:])
    for ax in ax1,ax2:
        # ax.set_ylim(0,53)
        # ax.legend(fontsize=labelfontsize-2)
        ax.tick_params('both',labelsize=labelfontsize-1.5,rotation=30)
        # ax.tick_params(direction="in", which="minor", length=3)
        # ax.tick_params(direction="in", which="major", length=5)
        # ax.grid(which="major", ls="dashed", dashes=(1, 4), lw=1.0, zorder=0)
        ax.grid(True, linestyle='-', alpha=0.3,lw=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
            # Filled markers (without landscape data)
            Line2D([0], [0], marker=markers_dict['Min.'], color='black', label='N-L',
            markerfacecolor='black', markersize=labelfontsize-1, linestyle='none'),
            # Line2D([0], [0], marker=markers['ts'], color='black', label='Transition state (Non-Landscape Models)',
            # markerfacecolor='black', markersize=labelfontsize-1, linestyle='none'),
            # Unfilled markers (with landscape data)
            Line2D([0], [0], marker=markers_dict['Min.'], color='black', label='L',
            markerfacecolor='none', markeredgecolor='black', markersize=labelfontsize-1, linestyle='none'),
            # Line2D([0], [0], marker=markers['ts'], color='black', label='Transition state (Landscape Models)',
            # markerfacecolor='none', markeredgecolor='black', markersize=labelfontsize-1, linestyle='none'),
    ]
    ax2.legend(handles=legend_elements, loc='right', fontsize=labelfontsize-1,bbox_to_anchor=(1,0.96))
    # ax2.legend(handles=legend_elements, loc='right', fontsize=labelfontsize-1,bbox_to_anchor=(0.99,0.35))

    return ax1,ax2


import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpecFromSubplotSpec

def plot_fig4_perfect_match_comparison_2(fig,n_unphysical_critical_points_plot_gridspec,labelfontsize):

    # molecules = ['ethanol', 'malonaldehyde', 'salicylic','azobenzene','paracetamol','aspirin']
    # models = ['dftb','ani2x','aimnet2','allegro','mace','nequip','mace-mp-0b3','so3lr']
    # types = ['non_altitude','altitude']
    molecules = ['salicylic']
    models = ['ani2x']
    types = ['non_altitude']
    all_results = []

    for molecule in molecules:
        for model in models:
            for model_type in types:
                if (model_type=='altitude' and model in ['aimnet2','ani2x','mace-mp-0b3','dftb','so3lr']):
                    continue
                try:
                    if model_type == 'non_altitude':
                        new_type = 'retrain' # just a naming convention
                    else:
                        new_type = 'altitude'

                    # print(molecule,model,model_type)
                    file = f'examples/salicylic_acid_ani2x/landscape_runs/analysis_exact-matches.pkl'
                    with open(file, 'rb') as f:
                        dft_ktn, ml_ktn, minima_correspondences, missing_ml_minima, statistics = pickle.load(f)
                        n_dft_min = dft_ktn.n_minima
                        n_dft_ts = len(list(dft_ktn.G.edges))
                        d_m = statistics['minima_distances']
                        d_ts = statistics['ts_distances']
                        de_m = statistics['minima_relative_energy_diffs']
                        de_m_absolute = statistics['minima_absolute_energy_diffs']
                        de_ts = statistics['ts_relative_energy_diffs']
                        de_ts_absolute = statistics['ts_absolute_energy_diffs']
                        mm = statistics['missing_minima']
                        mp = statistics['extra_minima']
                        tm = statistics['missing_ts']
                        tp = statistics['extra_ts']

                        d_m = round(np.mean(d_m),5)
                        d_ts = round(np.mean(d_ts),5)
                        de_m = round(np.sqrt(np.mean(np.array(de_m)**2)),4)
                        de_m_absolute = round(np.sqrt(np.mean(np.array(de_m_absolute)**2)),4)
                        de_ts = round(np.sqrt(np.mean(np.array(de_ts)**2)),4)
                        de_ts_absolute = round(np.sqrt(np.mean(np.array(de_ts_absolute)**2)),4)
                        entry={'Molecule':molecule,'Model':model,'Type':model_type,'Avg RMSD min. pairs':d_m,'RMSE E min. pairs (rel. to glob. mins.)':de_m,'RMSE E min. pairs (abs)':de_m_absolute,
                                'Extra mins.':mp,'Missing mins.':mm, 'DFT #min':n_dft_min,
                                'Avg. RMSD TS pairs':d_ts,'RMSE E TS pairs (rel. to glob. mins.)':de_ts,'RMSE E TS pairs (abs)':de_ts_absolute,
                                'Extra TS':tp,'Missing TS':tm, 'DFT #TS':n_dft_ts}
                        all_results.append(entry)

                except FileNotFoundError:
                    print(f'No output file for {model} {molecule} {type}.')
                    pass

    df = pd.DataFrame(all_results)
    df['Matched mins.']=df['DFT #min']-df['Missing mins.']
    df['Matched TS']=df['DFT #TS']-df['Missing TS']

    # Set academic style - clean and scientific
    plt.rcParams.update({
        # 'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.dpi': 150,
        'axes.axisbelow': True
    })

    # Your existing data processing code...
    model_order = ['ani2x']

    grouped = (
        df
        .groupby(["Model", "Type"])[["Extra mins.", "Extra TS"]]
        .sum()
        .reset_index()
    )
    # Reorder models
    grouped['Model_order'] = grouped['Model'].apply(lambda x: model_order.index(x) if x in model_order else 999)
    grouped = grouped.sort_values('Model_order').drop('Model_order', axis=1)
    # Pivot the data
    pivot_models = grouped.pivot_table(
        index="Model", 
        columns="Type", 
        values=["Extra mins.", "Extra TS"]
    )
    pivot_models = pivot_models.reindex(model_order)
    

    # Set academic style - clean and scientific
    plt.rcParams.update({
        # 'font.size': 11,
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.dpi': 150,
        'axes.axisbelow': True
    })

    # Create sub-gridspec for the first column (minima plot with axis break)
    minima_gs = GridSpecFromSubplotSpec(2, 1, n_unphysical_critical_points_plot_gridspec[0, 0], 
                                        height_ratios=[1, 2], hspace=0.05)
    
    # Create the broken axis subplots
    ax1_top = fig.add_subplot(minima_gs[0, 0])    # Upper part (60-70)
    ax1_bottom = fig.add_subplot(minima_gs[1, 0]) # Lower part (0-25)
    
    # Create the regular TS subplot
    ax2 = fig.add_subplot(n_unphysical_critical_points_plot_gridspec[0, 1])

    colors = {
        'non_altitude': {
            'min_physical': "#fb6767",     
            'min_nonphysical': "#ff9a9a",    
            'ts_nonphysical': "#ce3a3a",         
            'ts_physical': "#8d2626"    
        },
        'altitude': {
            'min_nonphysical': "#323e9a", 
            'min_physical': "#5761b0",
            'ts_physical': "#24329f",
            'ts_nonphysical': "#131b5b"       
        }
    }
    
    # Model labels for display
    display_labels = ['ANI2x']

    # Plotting function for broken axis
    def plot_broken_axis_bars(ax_top, ax_bottom, data_type, ylabel):
        x = np.arange(len(model_order))
        bar_width = 0.28
        data_type_color_label = 'ts' if data_type=='TS' else 'min'
        
        # Non-altitude bars (left)
        if (f'Extra {data_type}', 'non_altitude') in pivot_models.columns:
            extra_non_alt = pivot_models[(f'Extra {data_type}', 'non_altitude')].fillna(0)
            
            # Plot in bottom subplot (0-25)
            extra_non_alt_bottom = extra_non_alt.copy()
            extra_non_alt_bottom[extra_non_alt_bottom > 23] = 23
            
            ax_bottom.bar(x - bar_width/2, extra_non_alt_bottom, bar_width,
                label=f'N-L', 
                color=colors['non_altitude'][f'{data_type_color_label}_physical'], 
                alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Plot in top subplot (60+)
            extra_non_alt_top = extra_non_alt.copy()
            extra_non_alt_top[extra_non_alt_top <= 61] = 0
            
            ax_top.bar(x - bar_width/2, extra_non_alt_top, bar_width,
                color=colors['non_altitude'][f'{data_type_color_label}_physical'], 
                alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Altitude bars (right)
        if (f'Extra {data_type}', 'altitude') in pivot_models.columns:
            for i, model in enumerate(model_order):
                if not pd.isna(pivot_models.loc[model, (f'Extra {data_type}', 'altitude')]):
                    extra_alt = pivot_models.loc[model, (f'Extra {data_type}', 'altitude')]
                    
                    # if extra_alt <= 25:
                    # Plot in bottom subplot
                    ax_bottom.bar(i + bar_width/2, extra_alt, bar_width,
                        label=f'L' if i == len(model_order)-1 else "",
                        color=colors['altitude'][f'{data_type_color_label}_physical'], 
                        alpha=0.8, edgecolor='white', linewidth=0.5)
                    if extra_alt > 61:
                        # Plot in top subplot
                        ax_top.bar(i + bar_width/2, extra_alt, bar_width,
                            label=f'L' if i == len(model_order)-1 else "",
                            color=colors['altitude'][f'{data_type_color_label}_physical'], 
                            alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Plotting function for regular axis
    def plot_regular_bars(ax, data_type, ylabel):
        x = np.arange(len(model_order))
        bar_width = 0.28
        data_type_color_label = 'ts' if data_type=='TS' else 'min'
        
        # Non-altitude bars (left)
        if (f'Extra {data_type}', 'non_altitude') in pivot_models.columns:
            extra_non_alt = pivot_models[(f'Extra {data_type}', 'non_altitude')].fillna(0)
            
            ax.bar(x - bar_width/2, extra_non_alt, bar_width,
                label=f'N-L', 
                color=colors['non_altitude'][f'{data_type_color_label}_physical'], 
                alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Altitude bars (right)
        if (f'Extra {data_type}', 'altitude') in pivot_models.columns:
            for i, model in enumerate(model_order):
                if not pd.isna(pivot_models.loc[model, (f'Extra {data_type}', 'altitude')]):
                    extra_alt = pivot_models.loc[model, (f'Extra {data_type}', 'altitude')]
                    
                    ax.bar(i + bar_width/2, extra_alt, bar_width,
                        label=f'L' if i == len(model_order)-1 else "",
                        color=colors['altitude'][f'{data_type_color_label}_physical'], 
                        alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Plot the broken axis (minima)
    plot_broken_axis_bars(ax1_top, ax1_bottom, 'mins.', 'Count [$\longrightarrow$]')
    
    # Plot the regular axis (TS)
    plot_regular_bars(ax2, 'TS', 'Count [$\longrightarrow$]')
    
    # Format the broken axis
    ax1_top.set_ylim(61, 75)
    ax1_bottom.set_ylim(0, 23)
    
    # Remove spines between the two subplots
    ax1_top.spines['bottom'].set_visible(False)
    ax1_bottom.spines['top'].set_visible(False)
    ax1_top.tick_params(bottom=False, labelbottom=False)
    
    # Add break lines
    d = 0.01
    kwargs = dict(transform=ax1_top.transAxes, color='k', clip_on=False, linewidth=1)
    ax1_top.plot((-d, +d), (-d, +2*d), **kwargs)
    # ax1_top.plot((1-d, 1+d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax1_bottom.transAxes)
    ax1_bottom.plot((-d, +d), (1-d, 1+d), **kwargs)
    # ax1_bottom.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    # Set labels and ticks
    ax1_bottom.set_ylabel('Count [$\longleftarrow$]', fontsize=labelfontsize)
    ax1_bottom.set_xticks(np.arange(len(model_order)))
    ax1_bottom.set_xticklabels(display_labels, fontsize=labelfontsize-1.5, rotation=0)
    ax1_bottom.tick_params(labelsize=labelfontsize-1.5, rotation=30)
    
    ax1_top.tick_params(labelsize=labelfontsize-1.5)
    ax1_top.set_yticks([70])
    
    # Format the regular axis
    ax2.set_ylim(0, 72)
    ax2.set_ylabel('Count [$\longleftarrow$]', fontsize=labelfontsize)
    ax2.set_xticks(np.arange(len(model_order)))
    ax2.set_xticklabels(display_labels, fontsize=labelfontsize-1.5, rotation=0)
    ax2.tick_params(labelsize=labelfontsize-1.5, rotation=30)
    
    # Add legends
    ax1_bottom.legend(loc='upper left', fontsize=labelfontsize-1.5, framealpha=0.6, 
                     fancybox=True, shadow=False, ncol=2, bbox_to_anchor=(0.05,1.56))
    ax2.legend(loc='upper left', fontsize=labelfontsize-1.5, framealpha=0.6, 
              fancybox=True, shadow=False, ncol=2, bbox_to_anchor=(0.05,1))
    
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    # Grid styling
    for ax in [ax1_top, ax1_bottom, ax2]:
        ax.grid(True, linestyle='-', alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)

    print(f"\nDFT Reference: 28 minima, 67 transition states")
    return ax1_bottom, ax1_top,  ax2





# Create figure
width = 8
height = 7
labelfontsize=10
fig = plt.figure(figsize=(width, height))
fig_indexing = (_ for _ in ['A','B','C','D','E','F'])

# PERFECT MATCH KTN COMPARISON
width = 1
height = 0.24
left, top = 0,1
perfect_match_comparison_plot_gridspec = GridSpec(1, 2, left=left, right=left+width, bottom=top-height, top=top, figure=fig, hspace=0.05, wspace=0.3)
ax1,ax2 =  plot_fig4_perfect_match_comparison_1(fig,perfect_match_comparison_plot_gridspec, labelfontsize)
# ax1.text(-0.1,1.07,next(fig_indexing),weight='bold',transform=ax1.transAxes,fontsize=labelfontsize+6)
# Add panel labels
ax1.text(-0.2,1.15,next(fig_indexing),weight='bold',transform=ax1.transAxes,fontsize=labelfontsize+6)
ax1.text(-0.09,1.15,'# Missing Minima',weight='bold',transform=ax1.transAxes,fontsize=labelfontsize+1)
ax2.text(-0.2,1.15,next(fig_indexing),weight='bold',transform=ax2.transAxes,fontsize=labelfontsize+6)
ax2.text(-0.09,1.15,'# Missing TS',weight='bold',transform=ax2.transAxes,fontsize=labelfontsize+1)

# PERFECT MATCH KTN COMPARISON
width = 1
height = 0.24
left, top = 0,0.62
perfect_match_comparison_plot_gridspec = GridSpec(1, 2, left=left, right=left+width, bottom=top-height, top=top, figure=fig, hspace=0.05, wspace=0.3)
ax1,ax1_top,ax2 =  plot_fig4_perfect_match_comparison_2(fig,perfect_match_comparison_plot_gridspec, labelfontsize)
# ax1.text(-0.1,1.07,next(fig_indexing),weight='bold',transform=ax1.transAxes,fontsize=labelfontsize+6)
# Add panel labels
ax1_top.text(-0.2,1.42,next(fig_indexing),weight='bold',transform=ax1_top.transAxes,fontsize=labelfontsize+6)
ax1_top.text(-0.09,1.42,'# Additional Minima',weight='bold',transform=ax1_top.transAxes,fontsize=labelfontsize+1)
ax2.text(-0.2,1.15,next(fig_indexing),weight='bold',transform=ax2.transAxes,fontsize=labelfontsize+6)
ax2.text(-0.09,1.15,'# Additional TS',weight='bold',transform=ax2.transAxes,fontsize=labelfontsize+1)


# CLOSEST MATCH KTN COMPARISON
width = 1
height = 0.24
left, bottom = 0, 0.0
closest_match_RMSD_comparison_plot_gridspec = GridSpec(1, 2, left=left, right=left+width, bottom=bottom, top=bottom+height, figure=fig, hspace=0.05, wspace=0.3)
ax3,ax4 =  plot_fig4_closest_match_RMSD_comparison(fig,closest_match_RMSD_comparison_plot_gridspec, labelfontsize)
# Add panel labels
ax3.text(-0.2,1.15,next(fig_indexing),weight='bold',transform=ax3.transAxes,fontsize=labelfontsize+6)
ax3.text(-0.09,1.15,'Structural Similarity of Closest Min. Matches',weight='bold',transform=ax3.transAxes,fontsize=labelfontsize+1)
ax4.text(-0.2,1.15,next(fig_indexing),weight='bold',transform=ax4.transAxes,fontsize=labelfontsize+6)
ax4.text(-0.09,1.15,'Rel. Energy RMSE of Closest Min. Matches',weight='bold',transform=ax4.transAxes,fontsize=labelfontsize+1)

plt.savefig('examples/salicylic_acid_ani2x/landscape_runs/stationary_points_analysis.pdf', bbox_inches='tight')






####### KineticTransitionNetwork comparison

import matplotlib.pyplot as plt
import networkx as nx
import pickle
import numpy as np
from topsearch.plotting.network import barrier_reweighting
from collections import defaultdict


import matplotlib.patches as patches


def get_tuple_stats(data):
    stats = defaultdict(list)
    for i, tuple_val in enumerate(data):
        stats[tuple_val].append(i)
    return {k: len(v) for k, v in stats.items()}

pos = None

def plot_network_comparison(ax,reference_ktn,ml_ktn,minima_correspondences, seed=42, scaling_factor=2.0):
        """
        Plot a comparison of the two networks.
        
        Parameters
        ----------
        energy_range : list, optional
            Min and max energy for color scale, by default [-0.01, 0.01]
        seed : int, optional
            Random seed for layout, by default 42
        scaling_factor : float, optional
            Scaling factor for displacement vectors, by default 2.0
            
        Returns
        -------
        tuple
            (fig, ax) matplotlib objects for further customization
        """
        if minima_correspondences is None:
            raise ValueError("Must call find_minima_correspondences first")
        global pos 
        color_scheme = 'Blues'
        # Create weighted graph for better layout
        mapping = {0: 1, 1: 0, 2: 2, 3:3, 4:4,5:5,6:6}
        reference_ktn.G = nx.relabel_nodes(reference_ktn.G, mapping)
        new_minima_correspondence = []
        ml_ktn_mapping = {}
        for ref_idx,ml_idx,dist in minima_correspondences:
            new_minima_correspondence.append((mapping[ref_idx],ml_idx,dist))
        minima_correspondences = new_minima_correspondence

        print(minima_correspondences)
        g_weighted = barrier_reweighting(reference_ktn)
        if not pos:
            print(f'FOUND NOT POS, initializing')
            pos = nx.spring_layout(g_weighted, seed=seed)
        
        # # Remove self-loops if present
        # reference_ktn.G.remove_edges_from(nx.selfloop_edges(reference_ktn.G))
        
        n_atoms = reference_ktn.G.nodes[0]['coords'].shape[0]/3
        print(f'N atoms {n_atoms}')
        # Calculate colors based on energy relative to global minimum
        ref_min_energy = reference_ktn.min_e/n_atoms*1e3
        vmin = np.min([reference_ktn.G.nodes[i]['energy']/n_atoms*1e3 for i in reference_ktn.G.nodes]-ref_min_energy)
        vmax = np.max([reference_ktn.G.nodes[i]['energy']/n_atoms*1e3 for i in reference_ktn.G.nodes]-ref_min_energy)
        colors = np.array([
            reference_ktn.G.nodes[i]['energy']/n_atoms*1e3-ref_min_energy
            for i in reference_ktn.G.nodes
        ])
        
        # Draw reference network
        network_contours = nx.draw_networkx_nodes(
            reference_ktn.G, pos, 
            node_color=colors,
            cmap=plt.get_cmap(color_scheme), 
            ax=ax, 
            vmin=vmin, 
            vmax=vmax,
            edgecolors='k',
            node_size=120
        )
        labels_pos = {pos_key:pos_item-(0.299,0.01) for pos_key,pos_item in pos.items()}
        nx.draw_networkx_labels(reference_ktn.G, labels_pos, ax=ax,
                            font_size=labelfontsize-2,
                            font_family='sans-serif',
                            font_weight='bold')
            
        # DRAW MULTIPLE EDGES WHEN FINDING THEM
        edges_counts = get_tuple_stats(list(reference_ktn.G.edges()))
        for edge,count in edges_counts.items():
            pos_node_0 = pos[edge[0]]
            if edge[0]!=edge[1]:
                pos_node_1 = pos[edge[1]]
                edge_vector = pos_node_0 - pos_node_1
                perp_direction = np.array([1,-edge_vector[0]/edge_vector[1]])
                perp_direction /= np.linalg.norm(perp_direction)
                offset_factor = 0.038
                if count%2!=0:
                        offsets = [(-1)**i*i*offset_factor for i in range(count)]
                else: 
                        offsets = [offset_factor + (-1)**i*i*offset_factor for i in range(count)]
            else:
                offsets = [(-1)**i*i*offset_factor for i in range(count)]
            for offset in offsets:
                if not edge[0]==edge[1]:
                        new_pos = {edge[0]:pos_node_0 + perp_direction*offset, edge[1]:pos_node_1 + perp_direction*offset}
                else:
                        new_pos = {edge[0]:pos_node_0 + offset}
                nx.draw_networkx_edges(reference_ktn.G, new_pos, ax=ax, edgelist=[edge],width=1.2)

        # Create dictionary to store ML node positions
        np.random.seed(seed)
        ml_pos = {}
        skipped_nodes = []
        # Create random displacement vector scaled by distance
        direction = np.random.rand(2) # this is reproducible due to seed
        direction = direction / np.linalg.norm(direction)
        
        # Draw ML network nodes and connections to reference nodes
        for ref_idx, ml_idx, distance in minima_correspondences:
            
            offset = direction*0
            displacement = offset + direction * distance * scaling_factor
            
            # Calculate energy difference for coloring
            ml_energy_diff = np.abs(((ml_ktn.G.nodes[ml_idx]['energy']-ml_ktn.min_e)/n_atoms - (reference_ktn.G.nodes[ref_idx]['energy']-reference_ktn.min_e)/n_atoms)*1e3)
            # + \
             #               ref_min_energy - ml_ktn.G.nodes[minima_correspondences[0][1]]['energy']
            
            # Position ML node
            new_pos = np.array([pos[ref_idx][0] + displacement[0], pos[ref_idx][1] + displacement[1]])
            ml_pos[ml_idx] = new_pos
            
            # Draw connection between corresponding nodes
            ax.plot(
                [new_pos[0], pos[ref_idx][0]], 
                [new_pos[1], pos[ref_idx][1]], 
                ls='-', 
                c="#A9A9A9", 
                zorder=0.1
            )
            
            # Draw ML node
            cmap_for_colorbar=  ax.scatter(
                new_pos[0], 
                new_pos[1], 
                s=50, 
                c=ml_energy_diff, 
                cmap=plt.get_cmap('Purples'), 
                vmin=0, 
                vmax=15, 
                zorder=10, 
                edgecolors='gray'
            )

        # Draw missing DFT nodes
        matched_dft_minima = [i[0] for i in minima_correspondences]
        unmatched_dft_minima = [node for node in reference_ktn.G.nodes if node not in matched_dft_minima]
        unmatched_pos={}
        for unmatched_dft_min in unmatched_dft_minima:
            displacement = direction * 0.1
            new_pos = np.array([pos[unmatched_dft_min][0] + displacement[0], pos[unmatched_dft_min][1] + displacement[1]])

            # Instead of ax.scatter(x, y, edgecolor='red')
            circle = patches.Circle((new_pos[0], new_pos[1]), radius=0.048,  # adjust size
                                facecolor='white', 
                                edgecolor='red', 
                                linestyle=':',
                                linewidth=1.5,
                                zorder=10)
            ax.add_patch(circle)
            unmatched_pos[unmatched_dft_min]=new_pos
        
        # Draw ML network edges
        for ref_min1, ref_min2 in list(reference_ktn.G.edges()):
            if ref_min1 in skipped_nodes or ref_min2 in skipped_nodes:
                print(f"Skipping edge ({ref_min1}, {ref_min2}) due to threshold")
                continue
            
            ml_min1 = [ml_idx for r_idx, ml_idx, _ in minima_correspondences if r_idx == ref_min1]
            ml_min2 = [ml_idx for r_idx, ml_idx, _ in minima_correspondences if r_idx == ref_min2]
            edge_color = 'r'
            if ml_min1:
                ml_min1 = ml_min1.pop()
                x1 = ml_pos[ml_min1][0]
                y1 = ml_pos[ml_min1][1]
            else:
                x1 = unmatched_pos[ref_min1][0]
                y1 = unmatched_pos[ref_min1][1]

            n_ml_edges = 0

            if ml_min2:
                ml_min2 = ml_min2.pop()
                x2 = ml_pos[ml_min2][0]
                y2 = ml_pos[ml_min2][1]
                if ml_min1:
                    n_ml_edges = ml_ktn.G.number_of_edges(ml_min1, ml_min2)
            else:
                x2 = unmatched_pos[ref_min2][0]
                y2 = unmatched_pos[ref_min2][1]

            n_dft_edges = reference_ktn.G.number_of_edges(ref_min1,ref_min2)


            
            # DRAW MULTIPLE EDGES WHEN FINDING THEM
            pos_node_0 = np.array([x1,y1])
            if ref_min1!=ref_min2:
                pos_node_1 = np.array([x2,y2])
                edge_vector = pos_node_0 - pos_node_1
                perp_direction = np.array([1,-edge_vector[0]/edge_vector[1]])
                perp_direction /= np.linalg.norm(perp_direction)
                offset_factor = 0.045
                if n_dft_edges%2!=0:
                    offsets = [(-1)**i*i*offset_factor for i in range(max([n_ml_edges,n_dft_edges]))]
                else: 
                    offsets = [offset_factor + (-1)**i*i*offset_factor for i in range(max([n_ml_edges,n_dft_edges]))]
            else:
                offsets = [(-1)**i*i*offset_factor for i in range(count)]

            for offset in offsets:
                if n_ml_edges<=0:
                    edge_color='r'
                else: 
                    edge_color='k'
                n_ml_edges-=1
                if ref_min1!= ref_min2:
                    new_pos = {ref_min1:pos_node_0 + perp_direction*offset,ref_min2:pos_node_1 + perp_direction*offset}
                else: 
                    new_pos = {ref_min1:pos_node_0 + offset}
                edges = nx.draw_networkx_edges(reference_ktn.G, new_pos,
                                       ax=ax, edgelist=[(ref_min1, ref_min2)],
                                       edge_color=edge_color,style='dashed',node_size=200,width=1.5)

        return ax,network_contours,pos,cmap_for_colorbar

def plot_fig4_ktn_comparison(fig,ktn_comparison_plot_gridspec,ktn_comparison_plot_gridspec_left_legend,ktn_comparison_plot_gridspec_right_legend,labelfontsize,calc_type='retrain'):

    plt.rcdefaults()
    
    axs =[]
    for _ in range(1):
        ax = fig.add_subplot(ktn_comparison_plot_gridspec[0,_])
        axs.append(ax)
        
    global pos 
    
    import pickle 
    molecule = 'salicylic'
    models = ['ani2x']
    titles = ['ANI2x']
    vmin,vmax,hmin,hmax= 0,0,0,0
    for idx,(ax,model) in enumerate(zip(axs,models)):
        if calc_type=='altitude' and model in ['ani2x','aimnet2','dftb','mace-mp-0b3','so3lr']:
            ax.remove()
            continue
        print(calc_type,model)
        with open(f"examples/salicylic_acid_ani2x/landscape_runs/analysis_exact-matches.pkl", 'rb') as f:
            dft_ktn, ml_ktn, minima_correspondences, missing_ml_minima, _ = pickle.load(f)
        # reference_atoms = canonicalize_atoms(f"{dft_landscape_path}/{molecule}.xyz", dft_ktn)
        ax,_,pos,cmap_for_colorbar = plot_network_comparison(ax,dft_ktn,ml_ktn,minima_correspondences,scaling_factor=3,seed=2)
        if not pos:
            print(f'FOUND NOT POS IN MAIN LOOP')
            pos=pos
        ax.set_title(titles[idx],fontsize=labelfontsize)

        v = ax.get_ylim()
        h = ax.get_xlim()
        if v[0]<vmin:
            vmin = v[0]
        if v[1]>vmax:
            vmax = v[1]
        if h[0]<hmin:
            hmin=h[0]
        if h[1]>hmax:
            hmax=h[1]

    for ax in axs:
        ax.set_ylim(vmin-0.08,vmax+0.08)
        ax.set_xlim(hmin-0.32,hmax+0.039)

    if calc_type=='altitude':
        cbar_ax = fig.add_subplot(ktn_comparison_plot_gridspec_left_legend)
        cbar_ax2 = fig.add_subplot(ktn_comparison_plot_gridspec_right_legend)
        # Create colorbar in a new axis that doesn't affect existing subplots
        plt.colorbar(_, cax=cbar_ax, label='Rel. Energy\n(meV/atom)')#,labelsize=labelfontsize)
        cbar_ax.tick_params(right=False,left=True,labelright=False,labelleft=True,labelsize=labelfontsize)
        cbar_ax.yaxis.set_label_position("left")
        # Create colorbar in a new axis that doesn't affect existing subplots
        plt.colorbar(cmap_for_colorbar, cax=cbar_ax2, label='|$E^{rel}_{DFT}-E^{rel}_{ML}$|\n(meV/atom)')#,labelsize=labelfontsize)
        cbar_ax2.tick_params(labelsize=labelfontsize)

            
    import matplotlib.lines as mlines

    # Create proxy artists for the legend
    red_dashed = mlines.Line2D([], [], color='red', linestyle='--', 
                            label='Missing ML TS')
    black_dashed = mlines.Line2D([], [], color='black', linestyle='--', 
                                label='MLP TS')
    black_solid = mlines.Line2D([], [], color='black', linestyle='-', 
                                label='DFT TS')
    gray_solid = mlines.Line2D([], [], color="#A9A9A9", linestyle='-', 
                                label='Min. Match')
    black_circle = mlines.Line2D([], [], color='k', linestyle='',marker='o',markerfacecolor='white', 
                                label='DFT Min.')
    red_dashed_circle = patches.Circle((0,0), radius=1,  # adjust size
                                    facecolor='white', 
                                    edgecolor='red', 
                                    linestyle=':',
                                    linewidth=1.5,
                                    label='Missing MLP Min.')
    gray_circle = patches.Circle((0,0), radius=1,  # adjust size
                                    facecolor='white', 
                                    edgecolor='gray', 
                                    linestyle='-',
                                    linewidth=1.1,
                                    label='MLP Min.')

    from matplotlib.legend_handler import HandlerPatch
    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = patches.Circle(xy=center, radius=width//6)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p]

    if calc_type=='altitude':
        # If you want the legend outside the figure 
        legend = axs[3].legend(handles=[black_circle,gray_circle,black_solid,black_dashed,red_dashed_circle, red_dashed,gray_solid], 
                        loc='upper right',
                        bbox_to_anchor=(-1.3, 1.),  # bottom of figure
                        ncol=1,  # horizontal layout
                        frameon=True,
                        framealpha=0.9,
                        fontsize=labelfontsize-1,
                        handler_map={patches.Circle: HandlerCircle()})
        
    for ax in axs:
        # Alternative: Access through ax.collections
        all_collections = ax.collections
        print(f"Collections: {len(all_collections)}")

        # Set zorder for all collections
        for i, collection in enumerate(all_collections):
            collection.set_zorder(i + 1)
            print(f"Collection {i}: zorder = {collection.get_zorder()}")

    return axs 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
# Create figure
width = 2
height = 8.5
labelfontsize=10
fig = plt.figure(figsize=(width, height))
fig_indexing = (_ for _ in ['A','B','C','D','E','F'])



# SALICYLIC KTN COMPARISON
width = 1.10
legend_width=0.017
height = 0.185
left, bottom = -0.09, 0.9
ktn_comparison_plot_gridspec_left_legend = GridSpec(1, 1, left=left, right=left+legend_width-0.008, bottom=bottom, top=bottom+height,figure=fig)[0,0]
ktn_comparison_plot_gridspec_right_legend = GridSpec(1, 1, left=left + width +legend_width+0.008, right=left+width+2*legend_width, bottom=bottom, top=bottom+height,figure=fig)[0,0]
ktn_comparison_plot_gridspec = GridSpec(1, 1, left=left, right=left+width, bottom=bottom, top=bottom+height,figure=fig,hspace = 0.,wspace = 0.05)
ax5 = plot_fig4_ktn_comparison(fig,ktn_comparison_plot_gridspec,ktn_comparison_plot_gridspec_left_legend,ktn_comparison_plot_gridspec_right_legend,labelfontsize)
ax5=ax5[0]
ax5.text(0.02,1.19,next(fig_indexing),weight='bold',transform=ax5.transAxes,fontsize=labelfontsize+6)
ax5.text(0.35,1.19,'N-L KTN',weight='bold',transform=ax5.transAxes,fontsize=labelfontsize+1)


plt.savefig('/Users/vcarare/dev/mlp-landscapes/examples/salicylic_acid_ani2x/landscape_runs/ktn_comparison.pdf', bbox_inches='tight')