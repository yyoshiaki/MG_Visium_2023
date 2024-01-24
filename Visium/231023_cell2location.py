#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm import tqdm
import os

import cell2location

from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text for PDFs


# In[2]:


results_folder = '../scanpy/231023_cell2loc/'

# create paths and names to results folders for reference regression and cell2location models
ref_run_name = f'{results_folder}/reference_signatures'
run_name = f'{results_folder}/cell2location_map'
adata_file = f"{ref_run_name}/sc.h5ad"
sc.settings.figdir = '{}/graph'.format(results_folder)


# ## create ref

# In[3]:


adata_vis = sc.read('../scanpy/231018_combine_thymoma_norm/res.h5ad')
adata_vis.obs_names_make_unique()
adata_vis.X = adata_vis.layers['counts'].astype(int)
adata_vis


# In[4]:


adata_vis.var


# In[5]:


# find mitochondria-encoded (MT) genes
adata_vis.var['MT_gene'] = [gene.startswith('MT-') for gene in adata_vis.var.index]

# remove MT genes for spatial mapping (keeping their counts in the object)
adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]


# In[6]:


adata_ref = sc.read('/home/yyasumizu/media32TB/bioinformatics/MG_scRNAseq/scanpy/230713_combine_Xinetal/res_all_rmdoublet.h5ad')
# adata_ref.X = a[adata_ref.obs.index, adata_ref.var.index].X
adata_ref.X = adata_ref.layers['counts']


# In[7]:


adata_ref


# In[8]:


adata_ref.X.max()


# In[9]:


adata_ref.X = adata_ref.X.astype(int)


# In[10]:


from cell2location.utils.filtering import filter_genes
selected = filter_genes(adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)

# filter the object
adata_ref = adata_ref[:, selected].copy()


# In[11]:


# prepare anndata for the regression model
cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
                        # 10X reaction / sample / batch
                        batch_key='sample',
                        # cell type, covariate used for constructing signatures
                        labels_key='cluster_L2',
                        # multiplicative technical effects (platform, 3' vs 5', donor effect)
#                         categorical_covariate_keys=['']
                       )


# In[12]:


# create the regression model
from cell2location.models import RegressionModel
mod = RegressionModel(adata_ref)

# view anndata_setup as a sanity check
mod.view_anndata_setup()


# In[13]:


mod.train(max_epochs=250, use_gpu=True)


# In[14]:


mod.plot_history(20)


# In[15]:


# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_ref = mod.export_posterior(
    adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
)

# Save model
mod.save(f"{ref_run_name}", overwrite=True)

# Save anndata object with results
adata_file = f"{ref_run_name}/sc.h5ad"
adata_ref.write(adata_file)
adata_file


# In[16]:


mod.plot_QC()


# ## Resume

# In[20]:


adata_vis = sc.read('../scanpy/231018_combine_thymoma_norm/res.h5ad')
adata_vis.obs_names_make_unique()
adata_vis.X = adata_vis.layers['counts'].astype(int)
adata_vis


# In[21]:


adata_ref = sc.read_h5ad(adata_file)
mod = cell2location.models.RegressionModel.load(f"{ref_run_name}", adata_ref)


# In[22]:


if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]


# In[6]:


# find shared genes and subset both anndata and reference signatures
intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()


# In[147]:


for p in tqdm(adata_vis.obs['project'].unique()):
    
    mod = cell2location.models.RegressionModel.load(f"{ref_run_name}", adata_ref)
    
    if os.path.exists('{}/adata_infered_{}.h5ad'.format(results_folder, p)):
        print('skipped ', p)
        continue
    print(p)
    a = adata_vis[adata_vis.obs['project'] == p].copy()
    # 230212 added
    a.X = a.X.astype(int)
    sc.pp.filter_cells(a, min_genes=200)
    
    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=a, batch_key="sample_id")
    # create and train the model
    mod = cell2location.models.Cell2location(
        a, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=30,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20
    )
    mod.view_anndata_setup()

    mod.train(max_epochs=30000,
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1,
          use_gpu=True
         )

    # plot ELBO loss history during training, removing first 100 epochs from the plot
    mod.plot_history(1000)
    plt.legend(labels=['full data training']);
    
    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    a = mod.export_posterior(
        a, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
    )

    # Save model
#     mod.save(f"{run_name}", overwrite=True)

    # mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)
    
    mod.plot_QC()
    
    a.obs[a.uns['mod']['factor_names']] = a.obsm['q05_cell_abundance_w_sf']
    
    for s in a.obs['sample_id'].unique():
        sc.pl.spatial(a[a.obs['sample_id']==s], cmap='magma',
                  # show first 8 cell types
                  color=list(adata_ref.obs['minor_cluster'].cat.categories), 
                  ncols=4, size=1.3, spot_size=.15, 
    #                   img_key='hires',
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0, vmax='p99.2', library_id=s,
                   save='spatial_{}.pdf'.format(s)
                 )
        
    # Save anndata object with results
    a.write('{}/adata_infered_{}.h5ad'.format(results_folder, p))


# In[148]:


a.write('{}/adata_infered_{}.h5ad'.format(results_folder, p))