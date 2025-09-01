import pickle 
import os, os.path 
from ase.io import read 
import numpy as np 
import matplotlib.pyplot as plt

# To construct rate matrix
from scipy.sparse import issparse, diags

import PyGT
from ase.io import read 
import os, os.path 


def get_rate_properties(ktn, atoms, folder: str) -> None:
    """ Compute the rate properties for each stationary points and write
        to file: inertia tensor eigenvalues and average vibrational freq """

    min_energies = np.zeros((ktn.n_minima), dtype=float)
    min_ones = np.ones((ktn.n_minima), dtype=int)
    for i in range(ktn.n_minima):
        min_energies[i] = ktn.get_minimum_energy(i)
    ts_idxs = np.zeros((ktn.n_ts, 2), dtype=int)
    ts_energies = np.zeros((ktn.n_ts), dtype=float)
    ts_ones = np.ones((ktn.n_ts), dtype=float)
    c = 0
    for u, v, edge_idx in ktn.G.edges:
        ts_idxs[c] = [u+1, v+1]
        ts_energies[c] = ktn.get_ts_energy(u, v, edge_idx)
        c += 1

    out_min = np.vstack((min_energies, min_ones, min_ones, min_ones, min_ones, min_ones))
    out_ts = np.vstack((ts_energies, ts_ones, ts_ones, ts_idxs[:, 1], ts_idxs[:, 0], ts_ones, ts_ones, ts_ones))
    np.savetxt(f'{folder}/min.data', out_min.transpose())
    np.savetxt(f'{folder}/ts.data', out_ts.transpose())


molecule = 'salicylic'
models = ['ani2x']
titles = ['ANI2x']
calc_types = ['retrain']

atoms = read('examples/salicylic_acid_ani2x/data/salicylic_acid_ground_state_canon_perm.xyz')
for calc_type in calc_types:
    for idx,model in enumerate(models):
        if calc_type=='altitude' and model in ['ani2x','aimnet2','dftb','mace-mp-0b3','so3lr']:
            continue
        
        ml_landscape_path = f"examples/salicylic_acid_ani2x/landscape_runs"

        with open(f"examples/salicylic_acid_ani2x/landscape_runs/analysis_closest-matches.pkl", 'rb') as f:
            dft_ktn, ml_ktn, minima_correspondences, missing_ml_minima, _ = pickle.load(f)
            
        print(minima_correspondences)
        pathways_to_test = [(3,4)]

        for pathway_to_test in pathways_to_test:
            i, f = pathway_to_test
            folder_to_make = ml_landscape_path + f'/pygt{i}-{f}'
            matched_dft_mins = [t[0] for t in minima_correspondences]
            if i not in matched_dft_mins or f not in matched_dft_mins:
                continue
            minima_correspondences_dict = {t[0]:t[1] for t in minima_correspondences}
            corresponding_ml_mins = [minima_correspondences_dict[dft_min] for dft_min in [i,f]]
            corresponding_ml_mins=[corresponding_ml_mins[0]+1,corresponding_ml_mins[1]+1]

            print(corresponding_ml_mins)
            
            if not os.path.exists(folder_to_make):
                os.makedirs(folder_to_make)
            get_rate_properties(ml_ktn,atoms,folder_to_make)
            with open(folder_to_make+'/min.A','w') as fl:
                fl.write(f'1\n{corresponding_ml_mins[1]}')
            with open(folder_to_make+'/min.B','w') as fl:
                fl.write(f'1\n{corresponding_ml_mins[0]}')




atoms = read('examples/salicylic_acid_ani2x/data/salicylic_acid_ground_state_canon_perm.xyz')
for calc_type in calc_types:
    for idx,model in enumerate(models):
        if calc_type=='altitude' and model in ['ani2x','aimnet2','dftb','mace-mp-0b3','so3lr']:
            continue

        ml_landscape_path = f"examples/salicylic_acid_ani2x/landscape_runs"

        with open(f"examples/salicylic_acid_ani2x/landscape_runs/analysis_closest-matches.pkl", 'rb') as f:
            dft_ktn, ml_ktn, minima_correspondences, missing_ml_minima, _ = pickle.load(f)
        
        print(calc_type, model)
        print(minima_correspondences)
        # pathways_to_test = [(5,4),(3,4),(6,4)]
        pathways_to_test = [(3,5,4)]#,(6,4)]

        for pathway_to_test in pathways_to_test:
            i, m, f = pathway_to_test
            folder_to_make = ml_landscape_path + f'/pygt{i}-{f}-only'
            matched_dft_mins = [t[0] for t in minima_correspondences]
            if i not in matched_dft_mins or m not in matched_dft_mins or f not in matched_dft_mins:
                continue
            minima_correspondences_dict = {t[0]:t[1] for t in minima_correspondences}
            corresponding_ml_mins = [minima_correspondences_dict[dft_min] for dft_min in [i,m,f]]
            if ml_ktn.G.has_edge(minima_correspondences_dict[i],minima_correspondences_dict[f]):
                ml_ktn.remove_ts(minima_correspondences_dict[i],minima_correspondences_dict[f])
                print(f'Removed TS {minima_correspondences_dict[i],minima_correspondences_dict[f]}')
            print(corresponding_ml_mins, [i,m,f])
            ml_ktn.remove_minima([minimum for minimum in range(ml_ktn.n_minima) if minimum not in corresponding_ml_mins])
            # permutation = np.argsort(corresponding_ml_mins)
            permutation = [sorted(corresponding_ml_mins).index(i) for i in corresponding_ml_mins] # not equal to argsort
            print(permutation)
            corresponding_ml_mins=[permutation[0]+1,permutation[-1]+1]
            print(f'Final endpoints: ', corresponding_ml_mins)
            if not os.path.exists(folder_to_make):
                os.makedirs(folder_to_make)
            get_rate_properties(ml_ktn,atoms,folder_to_make)
            with open(folder_to_make+'/min.A','w') as fl:
                fl.write(f'1\n{corresponding_ml_mins[1]}')
            with open(folder_to_make+'/min.B','w') as fl:
                fl.write(f'1\n{corresponding_ml_mins[0]}')




def plot_fig4_rate_comparison(fig,rate_plot_gridspec,labelfontsize):

    plt.rcdefaults()
    
    ax2 = fig.add_subplot(rate_plot_gridspec[0,0])
    ax = fig.add_subplot(rate_plot_gridspec[0,1])

    
            
    # pathways_to_test = [(5,4)]#,(3,4),(6,4)]
    pathway_to_test = (3,4)
    model_colors_dict = {'Non-Landscape':{'ani2x':'#a089bd','aimnet2':'#7cd1b8','mace':'#4682b4', 'nequip':'#e07b91', 'allegro':'#a3b600', 'dftb':"#af4949",'mace-mp-0b3':"#004C8E",'so3lr':"#e9da0b"},
                    'Landscape': {'mace':'#2e597a', 'nequip':'#a04d64', 'allegro':'#6d7e00'}}


    invtemps = np.linspace(0.01, 40, 10)
    molecules = ['salicylic']
    models = ['ani2x']
    display_labels = ['ANI2x']
    calc_types = ['retrain']


    ax.axhline(1,ls='--',c='k',lw=1.1,zorder=-2)
    ax2.axhline(1,ls='--',c='k',lw=1.1,zorder=-2)

    i, f = pathway_to_test
    dft_mfpt_dict = {}
    for data_path, label in zip(['examples/salicylic_acid_ani2x/data/dft_ktn'],
                            ['default']):
        data_path = data_path + f'/pygt{i}-{f}'
        dft_mfpt_dict[label] = []
        for _, beta in enumerate(invtemps):
            B, K, tau, N, u, s, Emin, retained = PyGT.io.load_ktn(path=data_path,beta=beta,screen=False)
            A_vec, B_vec = PyGT.io.load_ktn_AB(data_path,retained)
            I_vec = ~(A_vec+B_vec)
            F = u - s/beta # free energy
            pi = np.exp(-beta * F) / np.exp(-beta * F).sum() # stationary distribution
            # K has no diagonal entries
            if issparse(K):
                Q = K - diags(1.0/tau)
            else:
                Q = K - np.diag(1.0/tau)
            moments = PyGT.stats.compute_passage_stats(A_vec,B_vec,pi,Q,dopdf=False)
            dft_mfpt_dict[label].append(moments[0])
            # variance.append(np.sqrt(moments[1]-moments[0]**2)/moments[0])
    # ax.plot(invtemps, dft_mfpt1, '--', color = 'k',label='DFT')

    try:

        for calc_type in calc_types:
            for molecule in molecules:
                for model_idx,model in enumerate(models):
                    if calc_type == 'altitude' and model in ['aimnet2','ani2x','mace-mp-0b3','dftb','so3lr','nequip']:
                        continue 
                    
                    
                    atoms = read(f'examples/salicylic_acid_ani2x/data/salicylic_acid_ground_state_canon_perm.xyz')

                    i, f = pathway_to_test
                    base_folder = f'examples/salicylic_acid_ani2x/landscape_runs/pygt3-4'

                    if not os.path.exists(base_folder):
                        continue

                    
                    data = np.zeros((4, len(invtemps)))
                    # try:
                    mfpt1=[]
                    variance=[]
                    for i, beta in enumerate(invtemps):

                        B, K, tau, N, u, s, Emin, retained = PyGT.io.load_ktn(path=base_folder,beta=beta,screen=False)
                        A_vec, B_vec = PyGT.io.load_ktn_AB(base_folder,retained)
                        I_vec = ~(A_vec+B_vec)
                        # print(f'States in A,I,B: {A_vec.sum(),I_vec.sum(),B_vec.sum()}')
                        F = u - s/beta # free energy
                        pi = np.exp(-beta * F) / np.exp(-beta * F).sum() # stationary distribution
                        # K has no diagonal entries
                        if issparse(K):
                            Q = K - diags(1.0/tau)
                        else:
                            Q = K - np.diag(1.0/tau)
                        moments = PyGT.stats.compute_passage_stats(A_vec,B_vec,pi,Q,dopdf=False)
                        mfpt1.append(moments[0])
                        variance.append(np.sqrt(moments[1]-moments[0]**2)/moments[0])
                        if i==0:
                            print(f'States in A,I,B: {A_vec.sum(),I_vec.sum(),B_vec.sum()}')
                    color_label = 'Landscape' if calc_type=='altitude' else 'Non-Landscape'
                    plot_label = 'L' if calc_type=='altitude' else 'N-L'
                    plot_marker = 'd'  if calc_type=='altitude' else '-o'
                    plot_marker = '-s' if model=='mace' and calc_type=='altitude' else plot_marker
                    markerfacecolor = 'None' if model !='allegro' or calc_type!='altitude' else model_colors_dict[color_label][model]
                    markeredgecolor = model_colors_dict[color_label][model] if model !='allegro' or calc_type!='altitude' else 'k'
                    markeredgewidth = 1 if model !='allegro' or calc_type!='altitude' else 0.7
                    zorder = 1 if model !='allegro' or calc_type!='altitude' else 2
                    markersize = 10 if model !='allegro' or calc_type!='altitude' else 5
                    

                    dft_mfpt1 = dft_mfpt_dict[model] if model in ['aimnet2','mace-mp-0b3','so3lr'] else dft_mfpt_dict['default']
                    mfpt1_ratio = np.array(mfpt1)/np.array(dft_mfpt1)
                
                    ax.plot(invtemps, mfpt1_ratio, plot_marker, markerfacecolor=markerfacecolor, lw=1.5,markersize=markersize, markeredgewidth=markeredgewidth, mec = markeredgecolor,
                    color = model_colors_dict[color_label][model],label=display_labels[model_idx] + f' {plot_label}', zorder=zorder)
    except:
        print('The KTN is likely not fully connected. Run longer landscape exploration runs.')            

    i, f = pathway_to_test
    dft_mfpt_dict = {}
    for data_path, label in zip(['examples/salicylic_acid_ani2x/data/dft_ktn'],
                            ['default']):
        data_path = data_path + f'/pygt{i}-{f}-only'
        dft_mfpt_dict[label] = []
        for _, beta in enumerate(invtemps):
            B, K, tau, N, u, s, Emin, retained = PyGT.io.load_ktn(path=data_path,beta=beta,screen=False)
            A_vec, B_vec = PyGT.io.load_ktn_AB(data_path,retained)
            I_vec = ~(A_vec+B_vec)
            F = u - s/beta # free energy
            pi = np.exp(-beta * F) / np.exp(-beta * F).sum() # stationary distribution
            # K has no diagonal entries
            if issparse(K):
                Q = K - diags(1.0/tau)
            else:
                Q = K - np.diag(1.0/tau)
            moments = PyGT.stats.compute_passage_stats(A_vec,B_vec,pi,Q,dopdf=False)
            dft_mfpt_dict[label].append(moments[0])



    try:
        for calc_type in calc_types:
            for molecule in molecules:
                for model_idx,model in enumerate(models):
                    if calc_type == 'altitude' and model in ['aimnet2','ani2x','mace-mp-0b3','dftb','so3lr','nequip']:
                        continue 
                    
                    atoms = read(f'examples/salicylic_acid_ani2x/data/salicylic_acid_ground_state_canon_perm.xyz')

                    i, f = pathway_to_test
                    base_folder = f'examples/salicylic_acid_ani2x/landscape_runs/pygt3-4-only'

                    if not os.path.exists(base_folder):
                        continue
                    
                    data = np.zeros((4, len(invtemps)))
                    # try:
                    mfpt1=[]
                    variance=[]
                    for i, beta in enumerate(invtemps):

                        B, K, tau, N, u, s, Emin, retained = PyGT.io.load_ktn(path=base_folder,beta=beta,screen=False)
                        A_vec, B_vec = PyGT.io.load_ktn_AB(base_folder,retained)
                        I_vec = ~(A_vec+B_vec)
                        # print(f'States in A,I,B: {A_vec.sum(),I_vec.sum(),B_vec.sum()}')
                        F = u - s/beta # free energy
                        pi = np.exp(-beta * F) / np.exp(-beta * F).sum() # stationary distribution
                        # K has no diagonal entries
                        if issparse(K):
                            Q = K - diags(1.0/tau)
                        else:
                            Q = K - np.diag(1.0/tau)
                        moments = PyGT.stats.compute_passage_stats(A_vec,B_vec,pi,Q,dopdf=False)
                        mfpt1.append(moments[0])
                        variance.append(np.sqrt(moments[1]-moments[0]**2)/moments[0])
                        if i==0:
                            print(f'States in A,I,B: {A_vec.sum(),I_vec.sum(),B_vec.sum()}')
                    color_label = 'Landscape' if calc_type=='altitude' else 'Non-Landscape'
                    plot_label = 'L' if calc_type=='altitude' else 'N-L'
                    plot_label = '' if model in ['mace-mp-0b3','so3lr','dftb','ani2x','aimnet2'] else plot_label
                    plot_marker = 'd'  if calc_type=='altitude' else '-o'
                    plot_marker = '-s' if model=='mace' and calc_type=='altitude' else plot_marker
                    markerfacecolor = 'None' if model !='allegro' or calc_type!='altitude' else model_colors_dict[color_label][model]
                    markeredgecolor = model_colors_dict[color_label][model] if model !='allegro' or calc_type!='altitude' else 'k'
                    markeredgewidth = 1 if model !='allegro' or calc_type!='altitude' else 0.7
                    zorder = 1 if model !='allegro' or calc_type!='altitude' else 2
                    markersize = 10 if model !='allegro' or calc_type!='altitude' else 5

                    dft_mfpt1 = dft_mfpt_dict[model] if model in ['aimnet2','mace-mp-0b3','so3lr'] else dft_mfpt_dict['default']
                    mfpt1_ratio = np.array(mfpt1)/np.array(dft_mfpt1)
                    ax2.plot(invtemps, mfpt1_ratio, plot_marker, markerfacecolor=markerfacecolor, lw=1.5,markersize=markersize, markeredgewidth=markeredgewidth, mec = markeredgecolor,
                    color = model_colors_dict[color_label][model],label=display_labels[model_idx] + f' {plot_label}', zorder=zorder)
    except:
        print('The graph in the 3-4 isolated case is likely not fully connected. Run longer landscape exploration runs.')



    # ax.set_xlabel('$1/T$')
    ax.set_ylabel('$MFPT_{MLP}$ / $MFPT_{DFT}$',fontsize=labelfontsize)
    # ax.legend(loc='upper left',bbox_to_anchor=(1,0.45))
    ax.set_title('Within KTN',y=0.86)#,weight='bold')
    ax.set_yscale('log')
    ax.grid(True, linestyle='-', alpha=0.15,lw=0.5,zorder=-1e10)
    ax.set_xlabel('$1/k_BT$ (eV$^{-1}$)',fontsize=labelfontsize)
    ax2.set_xlabel('$1/k_BT$ (eV$^{-1}$)',fontsize=labelfontsize)
    # ax2.set_ylabel(r'$\frac{MFPT_{DFT}}{MFPT_{MLP}}$')
    ax2.set_ylabel('$MFPT_{MLP}$ / $MFPT_{DFT}$',fontsize=labelfontsize)
    legend = ax2.legend(loc='lower left',bbox_to_anchor=(-0.01,-0.01),ncol=2,framealpha=0.35)
    from matplotlib.lines import Line2D
    dummy = Line2D([0], [0], color='k', lw=0)
    legend_handles = legend.legend_handles
    legend_handles.insert(3,dummy)
    ax2.legend(handles=legend_handles,ncol=2,framealpha=0.0,loc='lower left',bbox_to_anchor=(-0.01,-0.07),fontsize=labelfontsize-1)
    ax2.set_yscale('log')
    ax2.set_title('Isolated Pathway',y=0.86)#,weight='bold')
    ax.set_ylim(5e-2,5e10)
    ax2.set_ylim(5e-9,5e3)
    ax2.grid(True, linestyle='-', alpha=0.15,lw=0.5,zorder=-1e10)
    ax.tick_params(labelsize=labelfontsize)
    ax2.tick_params(labelsize=labelfontsize)
    # fig.suptitle('Mean First Passage Times\nBetween Nodes 3 to 4 in Salicylic KTN',y=0.97)

    return ax, ax2


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
# Create figure
width = 8
height = 8.5
labelfontsize=10
fig = plt.figure(figsize=(width, height))
fig_indexing = (_ for _ in ['A','B','C','D','E','F'])




# RATE COMPARISON
width = 0.92
height = 0.34
left, bottom = -0.0, 0.235
rate_comparison_plot_gridspec = GridSpec(1, 2, left=left, right=left+width, bottom=bottom, top=bottom+height,figure=fig,hspace = 0.,wspace = 0.4)
ax14,ax13 = plot_fig4_rate_comparison(fig,rate_comparison_plot_gridspec,labelfontsize)
ax13.text(-0.2,1.089,next(fig_indexing),weight='bold',transform=ax13.transAxes,fontsize=labelfontsize+6)
ax13.text(-0.1,1.089,r'Mean First Passage Time Comparison For Pathway 3 $\rightarrow$ 5 $\rightarrow$ 4',weight='bold',transform=ax13.transAxes,fontsize=labelfontsize+1)


plt.savefig('examples/salicylic_acid_ani2x/landscape_runs/meanfirstpassagetimes.pdf', bbox_inches='tight')