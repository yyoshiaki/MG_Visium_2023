#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce

import scvi
import torch
scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

import squidpy as sq
print(f"squidpy=={sq.__version__}")

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.stats.multitest as multi

from tqdm import tqdm

seed = 0
np.random.seed(seed)
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


# In[2]:


# sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, color_map='viridis', transparent=False, frameon=False)  # low dpi (dots per inch) yields small inline figures

import matplotlib as mpl
# 2 lines below solved the facecolor problem.
# mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
sc.settings.autosave = True
sc.logging.print_header()

version = '231106_scDRS_withcontrol_scvi'

import os
os.makedirs('../scanpy/{}'.format(version), exist_ok=True)

sc.settings.figdir = '../scanpy/{}/graph'.format(version)
sc.settings.cachedir = '../scanpy/{}/cache'.format(version)
# %config InlineBackend.figure_frmat = 'retina'

import os
os.makedirs('../scanpy/{}'.format(version), exist_ok=True)
os.makedirs(sc.settings.figdir, exist_ok=True)


# ## prep MAGMA
# 
# in `230519_thymus_scDRS.ipynb`

# ## scDRS 1st step
# 
# control data
# sol `/home/yyasumizu/visium_control`

# In[3]:


adata_thy = sc.read('../scanpy/231028_combine_thymoma_norm_scvi/res.h5ad')
df_clu = pd.read_csv('/home/yyasumizu/media32TB/bioinformatics/cellxgene_data/MG_thymoma/Yasumizu_et_al_2023/231028_visium_combined.scvi_annotations/231101.csv',
            comment='#', index_col=0)
adata_thy.obs['niche_annot'] = df_clu.loc[adata_thy.obs.index,'niche_annot']

adata_thy.obs_names_make_unique()
adata_thy.obs.index = adata_thy.obs['project'].astype(str) + '_' + adata_thy.obs.index
adata_thy.X = adata_thy.layers['counts']

adata_thy.obs = adata_thy.obs[['sample_id', 'leiden_scVI_0.7', 'niche_annot']]
adata_thy.obs['tissue'] = 'thymus'
adata_thy.var = adata_thy.var[[]]


# In[5]:


adata_ctrl = sc.read('../data/230523.visium_controls.h5ad')
adata_ctrl.obs_names_make_unique()
adata_ctrl.obs.index = adata_ctrl.obs['sample_id'].astype(str) + '_' + adata_ctrl.obs.index
adata_ctrl.obs['leiden_scVI_0.7'] = 'ctrl'
adata_ctrl.obs['niche_annot'] = 'ctrl'
adata_ctrl


# In[6]:
adata = sc.concat([adata_thy, adata_ctrl], uns_merge='unique')

# In[8]:


sc.pp.calculate_qc_metrics(adata, inplace=True)


# In[9]:


df_cov = adata.obs[['n_genes_by_counts']]
df_cov['const'] = 1
df_cov.index.name = 'cell_id'
# df_cov = pd.concat([df_cov, pd.get_dummies(adata.obs['project'])[['Helmi_et_al', 'Suo_et_al']]], axis=1)
df_cov.head()


# In[10]:


df_cov.to_csv('../scanpy/{}/cov.tsv'.format(version), sep='\t')


# In[11]:


adata.write('../scanpy/{}/res.raw.scdrs.h5ad'.format(version))


# In[12]:


sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sce.pp.magic(adata, name_list='all_genes', knn=5)
adata.write('../scanpy/{}/res.magic.scdrs.h5ad'.format(version))


# In[13]:


adata


# In[14]:


scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=['sample_id'],
    continuous_covariate_keys=['total_counts'],
)

model = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
model.view_anndata_setup()
model.train()

SCVI_NORMALIZED_KEY = "scvi_normalized"
adata.X = model.get_normalized_expression(library_size=10e4)


# In[15]:


adata.write('../scanpy/{}/res.scvi.scdrs.h5ad'.format(version))


# ### raw
# 
# sol
# 
# ```bash
# cd ~/MG_visium
# mkdir -p scanpy/231106_scDRS_withcontrol_scvi/scDRS
# mkdir -p scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic
# mkdir -p scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi
# 
# scdrs compute-score \
# --h5ad-file scanpy/231106_scDRS_withcontrol_scvi/res.raw.scdrs.h5ad \
# --h5ad-species human \
# --gs-file data/GWAS/MG.GCST90093061_buildGRCh38.magma.gs \
# --gs-species human \
# --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS \
# --flag-filter-data True \
# --flag-raw-count True \
# --n-ctrl 1000 \
# --flag-return-ctrl-raw-score False \
# --flag-return-ctrl-norm-score True
# ```
# 
# 231106_scDRS_batch.sh
# ```bash
# #!/bin/bash
# set -xe
# 
# # seq 0 14 | xargs -P15 -I{} bash 231106_scDRS_batch.sh {}
# 
# scdrs compute-score \
#     --h5ad-file ../scanpy/231106_scDRS_withcontrol_scvi/res.raw.scdrs.h5ad \
#     --h5ad-species human \
#     --gs-file ../data/gs_file/magma_10kb_top1000_zscore.75_traits.batch/batch${1}.gs \
#     --gs-species human \
#     --out-folder ../scanpy/231106_scDRS_withcontrol_scvi/scDRS \
#     --flag-filter-data True \
#     --flag-raw-count True \
#     --n-ctrl 1000 \
#     --flag-return-ctrl-raw-score False \
#     --flag-return-ctrl-norm-score True
# ```
# 
# ### impute
# ```bash
# cd ~/MG_visium
# 
# scdrs compute-score \
# --h5ad-file scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
# --h5ad-species human \
# --gs-file data/GWAS/MG.GCST90093061_buildGRCh38.magma.gs \
# --gs-species human \
# --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic \
# --flag-filter-data True \
# --flag-raw-count False \
# --n-ctrl 1000 \
# --flag-return-ctrl-raw-score False \
# --flag-return-ctrl-norm-score True
# 
# scdrs compute-score \
# --h5ad-file scanpy/231106_scDRS_withcontrol_scvi/res.scvi.scdrs.h5ad \
# --h5ad-species human \
# --gs-file data/GWAS/MG.GCST90093061_buildGRCh38.magma.gs \
# --gs-species human \
# --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi \
# --flag-filter-data True \
# --flag-raw-count True \
# --n-ctrl 1000 \
# --flag-return-ctrl-raw-score False \
# --flag-return-ctrl-norm-score True
# ```
# 
# 231106_scDRS_batch_impute.sh
# ```bash
# #!/bin/bash
# set -xe
# 
# # seq 0 14 | xargs -P15 -I{} bash 231106_scDRS_batch_impute.sh {}
# 
# scdrs compute-score \
#     --h5ad-file ../scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
#     --h5ad-species human \
#     --gs-file ../data/gs_file/magma_10kb_top1000_zscore.75_traits.batch/batch${1}.gs \
#     --gs-species human \
#     --out-folder ../scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic \
#     --flag-filter-data True \
#     --flag-raw-count False \
#     --n-ctrl 1000 \
#     --flag-return-ctrl-raw-score False \
#     --flag-return-ctrl-norm-score True
# ```
# 
# ### adjust prop
# ```bash
# scdrs compute-score \
# --h5ad-file scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
# --h5ad-species human \
# --gs-file data/GWAS/MG.GCST90093061_buildGRCh38.magma.gs \
# --gs-species human \
# --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic_adjproptis \
# --flag-filter-data True \
# --flag-raw-count False \
# --n-ctrl 1000 \
# --adj_prop tissue \
# --flag-return-ctrl-raw-score False \
# --flag-return-ctrl-norm-score True
# 
# scdrs compute-score \
# --h5ad-file scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
# --h5ad-species human \
# --gs-file data/GWAS/MG.GCST90093061_buildGRCh38.magma.gs \
# --gs-species human \
# --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_adjproptis \
# --flag-filter-data True \
# --flag-raw-count True \
# --n-ctrl 1000 \
# --adj_prop tissue \
# --flag-return-ctrl-raw-score False \
# --flag-return-ctrl-norm-score True
# ```
# 
# ## downstream
# 
# ```bash
# scdrs perform-downstream \
#     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.raw.scdrs.h5ad \
#         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS/@.full_score.gz \
#         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS/ \
#         --group-analysis tissue \
#         --flag-filter-data True \
#         --flag-raw-count True
# 
#         
# scdrs perform-downstream \
#     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
#         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic/@.full_score.gz \
#         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic/ \
#         --group-analysis tissue \
#         --flag-filter-data True \
#         --flag-raw-count False
# 
#                 
# scdrs perform-downstream \
#     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.scvi.scdrs.h5ad \
#         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi/@.full_score.gz \
#         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi/ \
#         --group-analysis tissue \
#         --flag-filter-data True \
#         --flag-raw-count True
#    
# # downstream using all spots including controls
# scdrs perform-downstream \
#     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
#         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic/@.full_score.gz \
#         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic/ \
#         --group-analysis leiden_scVI_0.7 \
#         --flag-filter-data True \
#         --flag-raw-count False
# 
# 
# scdrs perform-downstream \
#     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
#         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic/@.full_score.gz \
#         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic/ \
#         --group-analysis niche_annot \
#         --flag-filter-data True \
#         --flag-raw-count False
# 
# scdrs perform-downstream \
#     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.scvi.scdrs.h5ad \
#         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi/@.full_score.gz \
#         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi/ \
#         --group-analysis leiden_scVI_0.7 \
#         --flag-filter-data True \
#         --flag-raw-count False
# 
# 
# scdrs perform-downstream \
#     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.scvi.scdrs.h5ad \
#         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi/@.full_score.gz \
#         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_scvi/ \
#         --group-analysis niche_annot \
#         --flag-filter-data True \
#         --flag-raw-count False
# 
# # # downstream only using thymus spots
# # scdrs perform-downstream \
# #     --h5ad-file ./scanpy/231106_thymus_scDRS_scvi/res.scdrs.h5ad \ # specifying only thumus
# #         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_adjproptis/@.full_score.gz \
# #         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_adjproptis \
# #         --group-analysis clusters_merge \
# #         --flag-filter-data True \
# #         --flag-raw-count True
#         
# # scdrs perform-downstream \
# #     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
# #         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic_adjproptis/@.full_score.gz \
# #         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic_adjproptis/ \
# #         --group-analysis tissue \
# #         --flag-filter-data True \
# #         --flag-raw-count False
# 
# # scdrs perform-downstream \
# #     --h5ad-file ./scanpy/231106_thymus_scDRS_scvi/res.magic.scdrs.h5ad \
# #         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic_adjproptis/@.full_score.gz \
# #         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic_adjproptis \
# #         --group-analysis clusters_merge \
# #         --flag-filter-data True \
# #         --flag-raw-count False
# 
# # scdrs perform-downstream \
# #     --h5ad-file ./scanpy/231106_scDRS_withcontrol_scvi/res.magic.scdrs.h5ad \
# #         --score-file scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic_adjproptis/MG.full_score.gz \
# #         --out-folder scanpy/231106_scDRS_withcontrol_scvi/scDRS_magic_adjproptis_allspots/ \
# #         --group-analysis clusters_merge \
# #         --flag-filter-data True \
# #         --flag-raw-count False
# ```

# ## downstram

# In[3]:


adata = sc.read('../scanpy/{}/res.raw.scdrs.h5ad'.format(version))


# In[4]:


adata


# In[5]:


dict_trait = {'MG': 'MG'}


# In[6]:


for trait, f in dict_trait.items():
    # adata.obs[trait] = pd.read_csv('../scanpy/{}/scDRS/{}.full_score.gz'.format(version, f), sep='\t', index_col=0)['norm_score']
    # adata.obs[trait + '_pval'] = pd.read_csv('../scanpy/{}/scDRS/{}.full_score.gz'.format(version, f), sep='\t', index_col=0)['pval']
    adata.obs[trait + '_magic'] = pd.read_csv('../scanpy/{}/scDRS_magic/{}.full_score.gz'.format(version, f), sep='\t', index_col=0)['norm_score']
    adata.obs[trait + '_magic_pval'] = pd.read_csv('../scanpy/{}/scDRS_magic/{}.full_score.gz'.format(version, f), sep='\t', index_col=0)['pval']
    # adata.obs[trait + '_magic_adjproptis'] = pd.read_csv('../scanpy/{}/scDRS_magic_adjproptis/{}.full_score.gz'.format(version, f), sep='\t', index_col=0)['norm_score']
    # adata.obs[trait + '_magic_pval_adjproptis'] = pd.read_csv('../scanpy/{}/scDRS_magic_adjproptis/{}.full_score.gz'.format(version, f), sep='\t', index_col=0)['pval']

sc.pl.umap(
    adata,
    color=dict_trait.keys(),
    color_map="BrBG_r",
    vmin=-5,
    vmax=5,
    s=20,
    save='scDRS'
)
# In[20]:


for s in adata.obs['sample_id'].unique():
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.spatial(adata[adata.obs.sample_id==s], img_key="lowres",
              color=['MG_magic'], cmap="BrBG_r",
                  vmin=-3, vmax=3, library_id=s,
                 save=f"{s}_scDRS_MG_magic.pdf")

for s in adata.obs['sample_id'].unique():
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.spatial(adata[adata.obs.sample_id==s], img_key="hires",
              color=dict_trait.keys(), cmap="BrBG_r",
                  vmin=-3, vmax=3, library_id=s,
                 save=f"{s}_scDRS.pdf")for s in adata.obs['sample_id'].unique():
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.spatial(adata[adata.obs.sample_id==s], img_key="hires",
              color=['MG_magic_adjproptis'], cmap="BrBG_r",
                  vmin=-3, vmax=3, library_id=s,
                 save=f"{s}_scDRS.pdf")
# In[22]:


sc.pl.violin(adata, keys='MG_magic', groupby='tissue', rotation=90)



# In[12]:


adata.obs.sort_values('MG_pval').head(50)


# In[23]:


import scdrs


# In[35]:


dict_df_stats = {
    trait: pd.read_csv(f"../scanpy/{version}/scDRS_magic/{trait}.scdrs_group.niche_annot", sep="\t", index_col=0)
    for trait in dict_trait.values()
}
dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['niche_annot'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        trait: df_stats.rename(index=dict_celltype_display_name)
        for trait, df_stats in dict_df_stats.items()
    },
    plot_kws={
        "vmax": 0.2,
        "cb_fraction":0.12
    }
)


# In[60]:


df_stats = pd.read_csv(f"../scanpy/{version}/scDRS_magic/{trait}.scdrs_group.niche_annot", sep="\t", index_col=0)
df_stats['n_fdr_0.1'] = df_stats['n_fdr_0.2']
df_stats = df_stats.loc[['cortex', 'junction', 'medulla', 'medulla_FN1', 'medulla_GC',
       'stroma', 'ctrl']]

dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['niche_annot'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        'MG': df_stats.rename(index=dict_celltype_display_name)
    },
    plot_kws={
        "vmax": 0.2,
        "cb_fraction":0.12
    }
)
cb = ax.collections[0].colorbar
cb.ax.set_title("Prop. of sig. cells (FDR < 0.2)", fontsize=8)

plt.savefig(str(sc.settings.figdir) + '/scDRS_magic_niche_annot.pdf', bbox_inches='tight')


# In[28]:


dict_df_stats['MG']


# In[54]:


dict_df_stats = {
    trait: pd.read_csv(f"../scanpy/{version}/scDRS_magic/{trait}.scdrs_group.leiden_scVI_0.7", sep="\t", index_col=0)
    for trait in dict_trait.values()
}
dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['tissue'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        trait: df_stats.rename(index=dict_celltype_display_name)
        for trait, df_stats in dict_df_stats.items()
    },
    plot_kws={
        "vmax": 0.2,
        "cb_fraction":0.12
    }
)


# In[61]:


df_stats = pd.read_csv(f"../scanpy/{version}/scDRS_magic/{trait}.scdrs_group.leiden_scVI_0.7", sep="\t", index_col=0)
df_stats['n_fdr_0.1'] = df_stats['n_fdr_0.2']
df_stats = df_stats.loc[['0', '1', '2', '3', '4',
       '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', 'ctrl']]

dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['niche_annot'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        'MG': df_stats.rename(index=dict_celltype_display_name)
    },
    plot_kws={
        "vmax": 0.2,
        "cb_fraction":0.12
    }
)
cb = ax.collections[0].colorbar
cb.ax.set_title("Prop. of sig. cells (FDR < 0.2)", fontsize=8)

plt.savefig(str(sc.settings.figdir) + '/scDRS_magic_leiden_scVI_0.7.pdf', bbox_inches='tight')


# In[62]:


df_stats


# In[71]:


df_stats = pd.read_csv(f"../scanpy/{version}/scDRS_magic/{trait}.scdrs_group.tissue", sep="\t", index_col=0)
df_stats['n_fdr_0.1'] = df_stats['n_fdr_0.2']
# df_stats = df_stats.loc[['0', '1', '2', '3', '4',
#        '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', 'ctrl']]

dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['tissue'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        'MG': df_stats.rename(index=dict_celltype_display_name)
    },
    plot_kws={
        "cb_vmax": 0.1,
        "cb_fraction":0.12
    }
)
cb = ax.collections[0].colorbar
cb.ax.set_title("Prop. of sig. cells (FDR < 0.2)", fontsize=8)

plt.savefig(str(sc.settings.figdir) + '/scDRS_magic_tissue.pdf', bbox_inches='tight')


# In[67]:


df_stats


# In[ ]:


dict_df_stats = {
    trait: pd.read_csv(f"../scanpy/{version}/scDRS_adjproptis/{trait}.scdrs_group.clusters_merge", sep="\t", index_col=0)
    for trait in dict_trait.values()
}
dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['clusters_merge'].unique() # adata.obs['clusters_merge'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        trait: df_stats.rename(index=dict_celltype_display_name)
        for trait, df_stats in dict_df_stats.items()
    },
    plot_kws={
        "vmax": 0.2,
        "cb_fraction":0.12
    }
)


# In[ ]:


dict_df_stats = {
    trait: pd.read_csv(f"../scanpy/{version}/scDRS_magic_adjproptis/{trait}.scdrs_group.tissue", sep="\t", index_col=0)
    for trait in dict_trait.values()
}
dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['tissue'].unique() # adata.obs['clusters_merge'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        trait: df_stats.rename(index=dict_celltype_display_name)
        for trait, df_stats in dict_df_stats.items()
    },
    plot_kws={
        "vmax": 0.2,
        "cb_fraction":0.12
    }
)


# In[ ]:


dict_df_stats = {
    trait: pd.read_csv(f"../scanpy/{version}/scDRS_magic_adjproptis/{trait}.scdrs_group.clusters_merge", sep="\t", index_col=0)
    for trait in dict_trait.values()
}
dict_celltype_display_name =  {
    x: x.replace('_', ' ') for x in adata.obs['clusters_merge'].unique() # adata.obs['clusters_merge'].cat.categories
}

fig, ax = scdrs.util.plot_group_stats(
    dict_df_stats={
        trait: df_stats.rename(index=dict_celltype_display_name)
        for trait, df_stats in dict_df_stats.items()
    },
    plot_kws={
        "vmax": 0.2,
        "cb_fraction":0.12
    }
)


# In[ ]:


dict_df_stats['MG']


# In[ ]:


multi.multipletests(dict_df_stats['MG']['assoc_mcp'], method='fdr_bh')

