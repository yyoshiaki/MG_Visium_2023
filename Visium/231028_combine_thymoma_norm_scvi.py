#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import squidpy as sq

import scvi
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.stats.multitest as multi

seed = 0
np.random.seed(seed)
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import shutil


# In[2]:


sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, color_map='viridis', transparent=False, frameon=False)  # low dpi (dots per inch) yields small inline figures

import matplotlib as mpl
# 2 lines below solved the facecolor problem.
# mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
sc.settings.autosave = True
sc.logging.print_header()
print(f"squidpy=={sq.__version__}")
torch.set_float32_matmul_precision("high")

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)

version = '231028_combine_thymoma_norm_scvi'

results_file = '../scanpy/{}/res.h5ad'.format(version)
results_magic_file = '../scanpy/{}/res.magic.h5ad'.format(version)
results_file_cellxgene = '../scanpy/{}/res.cxg.h5ad'.format(version)
save_dir = '../scanpy/{}/'.format(version)

import os
os.makedirs('../scanpy/{}'.format(version), exist_ok=True)

sc.settings.figdir = '../scanpy/{}/graph'.format(version)
sc.settings.cachedir = '../scanpy/{}/cache'.format(version)
# %config InlineBackend.figure_format = 'retina'

import os
os.makedirs('../scanpy/{}'.format(version), exist_ok=True)
os.makedirs(sc.settings.figdir, exist_ok=True)


# In[3]:


list_adata = []
a = sc.read('../scanpy/231028_MG_visium_prep_3/res.h5ad')
a.obs['clusters_orig'] = 'T_' + a.obs['clusters_thymoma'].astype(str)
del a.raw
a.X = a.layers['counts']

list_adata.append(a)

a = sc.read('../scanpy/230518_thymus_visium_publicdata/res.h5ad')
a.obs['clusters_orig'] = 'N_' + a.obs['clusters_normal'].astype(str)
del a.raw
a.X = a.layers['counts']
list_adata.append(a)


# In[4]:


list_adata


# In[5]:


adata = sc.concat(list_adata, uns_merge='unique', join='inner')
adata.obs_names_make_unique()
adata.obs.index = adata.obs['project'].astype(str) + '_' + adata.obs.index


# In[6]:


# drop spot without obsm attribute
spatial_data = adata.obsm['spatial']
print(np.isnan(spatial_data).sum())

valid_indices = ~np.isnan(spatial_data).any(axis=1)
adata = adata[valid_indices].copy()


# In[7]:


adata


# In[8]:


adata.layers['counts'].shape


# In[9]:


adata.layers['counts'] = adata.layers['counts'].astype(int)


# In[10]:


sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
adata.raw = adata

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    subset=False,
    layer="counts",
    flavor="seurat_v3",
    batch_key="sample_id"
)


# In[11]:


scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=['sample_id', "project"],
    continuous_covariate_keys=['total_counts'],
)


# In[12]:


model = scvi.model.SCVI(adata, n_layers=2, n_latent=30)


# In[13]:


model.view_anndata_setup()


# In[14]:


# model.train(plan_kwargs={"lr": 0.005})
model.train()


# In[15]:


model_dir = os.path.join(save_dir, "scvi_model")
model.save(model_dir, overwrite=True)


# In[16]:


SCVI_LATENT_KEY = "X_scVI"

latent = model.get_latent_representation()
adata.obsm[SCVI_LATENT_KEY] = latent
latent.shape

SCVI_NORMALIZED_KEY = "scvi_normalized"

adata.layers[SCVI_NORMALIZED_KEY] = model.get_normalized_expression(library_size=10e4)


# In[25]:


# use scVI latent space for UMAP generation
sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
sc.tl.umap(adata, min_dist=0.5, spread=2)
# neighbors were already computed using scVI
SCVI_CLUSTERS_KEY = "leiden_scVI"
sc.tl.leiden(adata, key_added=SCVI_CLUSTERS_KEY, resolution=1)


# In[26]:


sc.tl.embedding_density(adata, basis='umap', groupby='project')
sc.pl.embedding_density(
    adata, basis='umap', key='umap_density_project'
)


# In[27]:


sc.tl.embedding_density(adata, basis='umap', groupby='condition')
sc.pl.embedding_density(
    adata, basis='umap', key='umap_density_condition'
)


# In[35]:


markers = ['EPCAM', 'CD79A', 'PRDM1', 'MZB1', 'CXCL13', "CD4", 'FOXP3', 'AIRE', 'KRT6A', 'KRT15', 'KRT7', 'KRT17', 'DSG1', 'PSMB11', 
           'RAG1', 'DNTT', 'CIITA', 'CD80', 'CD86', 'FN1', 'CHRNA1', 'CHRNA3', 'GABRA5', 'NEFL', 'NEFM', 
           'KCNC2', 'MYH14', 'GRIN2A',
          'NR4A1', 'CXCR4', 'CXCL12', 'CXCL14', 'LEP', 'CD99']


# In[36]:


plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=markers, wspace=0.4,
          save='markers.pdf')


# In[184]:


plt.rcParams["figure.figsize"] = (4, 4)
sc.pl.umap(adata, color=['MKI67'], wspace=0.4,
          save=False)


# In[147]:


# neighbors were already computed using scVI
resolution=1.1
SCVI_CLUSTERS_KEY = f"leiden_scVI_{resolution}"
sc.tl.leiden(adata, key_added=SCVI_CLUSTERS_KEY, resolution=resolution)


# In[148]:


sc.pl.umap(
    adata,
    color=[SCVI_CLUSTERS_KEY],
    frameon=False,
)


# In[80]:


for s in adata.obs['sample_id'].unique():
    # plt.rcParams["figure.figsize"] = (4, 4)
    # sc.pl.spatial(adata[adata.obs['sample_id']==s], img_key="hires",
    #           color=["total_counts", "n_genes_by_counts"], library_id=s, cmap='Blues',
    #              save=f"{s}_QC.pdf")
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.spatial(adata[adata.obs['sample_id']==s], img_key="lowres",
              color=[SCVI_CLUSTERS_KEY],
              library_id=s,
                 save=f"{s}_{SCVI_CLUSTERS_KEY}.pdf")


# In[81]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata, color=SCVI_CLUSTERS_KEY, add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title=SCVI_CLUSTERS_KEY, save=f'outline_{SCVI_CLUSTERS_KEY}.pdf')


# In[82]:


sq.gr.spatial_neighbors(adata, library_key='sample_id', spatial_key='spatial')
sq.gr.nhood_enrichment(adata, cluster_key=SCVI_CLUSTERS_KEY)
sq.pl.nhood_enrichment(adata, cluster_key=SCVI_CLUSTERS_KEY)


# In[141]:


de_df = model.differential_expression(
    groupby=SCVI_CLUSTERS_KEY,
)
de_df.head()


# In[155]:


markers = {}
cats = adata.obs[SCVI_CLUSTERS_KEY].cat.categories
for i, c in enumerate(cats):
    cid = f"{c} vs Rest"
    cell_type_df = de_df.loc[de_df.comparison == cid]

    cell_type_df = cell_type_df[cell_type_df.lfc_mean > 0]

    cell_type_df = cell_type_df[cell_type_df["bayes_factor"] > 2]
    cell_type_df = cell_type_df[cell_type_df["non_zeros_proportion1"] > 0.1]

    markers[c] = cell_type_df.index.tolist()[:5]
    
markers


# In[156]:


sc.tl.dendrogram(adata, groupby=SCVI_CLUSTERS_KEY, use_rep="X_scVI")
sc.pl.dotplot(
    adata,
    markers,
    groupby=SCVI_CLUSTERS_KEY,
    dendrogram=False,
    color_map="Blues",
    swap_axes=True,
    use_raw=True,
    standard_scale="var",
    save=f'markers_{SCVI_CLUSTERS_KEY}'
)


# In[86]:


with plt.rc_context({"figure.figsize": (6, 3)}):
    sc.pl.violin(adata, keys=['score_yellow'], groupby=SCVI_CLUSTERS_KEY, rotation=90, 
                     save=f'score_yellow_{SCVI_CLUSTERS_KEY}')


# In[88]:


adata.obs['sample_id'].unique()


# In[170]:


s = 'H17040431'
sc.pl.spatial(adata[adata.obs['sample_id']==s], img_key="lowres",
          color=['CCR7', 'CCL19', 'CCL21', 'CCR9', 'CCL25', 
                 'CXCR4', 'CXCL12', 'CXCL14', 'CXCR5', 'CXCL13', 'CCR4', 'CCL17', 'CCL22', 
                 'CCR5', 'CCL3', 'CCL4', 'CXCR3', 'CXCL10', 'S1PR1', 'S1PR2'],
          library_id=s,
             save=False)


# In[171]:


s = 'H2002145-2_1'
sc.pl.spatial(adata[adata.obs['sample_id']==s], img_key="lowres",
          color=['CCR7', 'CCL19', 'CCL21', 'CCR9', 'CCL25', 
                 'CXCR4', 'CXCL12', 'CXCL14', 'CXCR5', 'CXCL13', 'CCR4', 'CCL17', 'CCL22', 
                 'CCR5', 'CCL3', 'CCL4', 'CXCR3', 'CXCL10', 'S1PR1', 'S1PR2'],
          library_id=s,
             save=False)


# In[172]:


s = 'GSM6281324_S2_A1'
sc.pl.spatial(adata[adata.obs['sample_id']==s], img_key="lowres",
          color=['CCR7', 'CCL19', 'CCL21', 'CCR9', 'CCL25', 
                 'CXCR4', 'CXCL12', 'CXCL14', 'CXCR5', 'CXCL13', 'CCR4', 'CCL17', 'CCL22', 
                 'CCR5', 'CCL3', 'CCL4', 'CXCR3', 'CXCL10', 'S1PR1', 'S1PR2'],
          library_id=s,
             save=False)


# In[108]:


adata


# In[113]:


adata.raw.to_adata().write(results_file_cellxgene)


# In[208]:


df_annot = pd.read_csv('/home/yyasumizu/media32TB/bioinformatics/cellxgene_data/MG_thymoma/Yasumizu_et_al_2023/231028_visium_combined.scvi_annotations/231101.csv',
            comment='#', index_col=0)
df_annot.head()


# In[209]:


adata


# In[199]:


adata.obs = adata.obs.drop(['niche_annot'], axis=1)
# adata.obs = adata.obs.drop(['niche_annot_x', 'niche_annot_y'], axis=1)


# In[211]:


adata.obs = pd.merge(adata.obs, df_annot, left_index=True, right_index=True, how='left')
adata.obs['niche_annot'] = adata.obs['niche_annot'].astype(str).astype('category')


# In[213]:


sc.pl.umap(
    adata,
    color=['niche_annot_v2'],
    frameon=False,
    save='niche_annot_v2'
)


# In[214]:


for s in adata.obs['sample_id'].unique():
    # plt.rcParams["figure.figsize"] = (4, 4)
    # sc.pl.spatial(adata[adata.obs['sample_id']==s], img_key="hires",
    #           color=["total_counts", "n_genes_by_counts"], library_id=s, cmap='Blues',
    #              save=f"{s}_QC.pdf")
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.spatial(adata[adata.obs['sample_id']==s], img_key="lowres",
              color=['niche_annot_v2'],
              library_id=s,
                 save=f"{s}_niche_annot_v2.pdf")


# In[215]:


with plt.rc_context({"figure.figsize": (6, 3)}):
    sc.pl.violin(adata, keys=['score_yellow'], groupby='niche_annot', rotation=90, 
                     save=f'score_yellow_niche_annot')


# In[216]:


with plt.rc_context({"figure.figsize": (6, 3)}):
    sc.pl.violin(adata, keys=['score_yellow'], groupby='niche_annot_v2', rotation=90, 
                     save=f'score_yellow_niche_annot_v2')


# In[204]:


adata


# In[207]:


sq.gr.spatial_neighbors(adata, library_key='sample_id', spatial_key='spatial')
sq.gr.nhood_enrichment(adata, cluster_key='niche_annot')
sq.pl.nhood_enrichment(adata, cluster_key='niche_annot', save='niche_annot.pdf')


# In[217]:


sq.gr.spatial_neighbors(adata, library_key='sample_id', spatial_key='spatial')
sq.gr.nhood_enrichment(adata, cluster_key='niche_annot_v2')
sq.pl.nhood_enrichment(adata, cluster_key='niche_annot_v2', save='niche_annot_v2.pdf')


# In[167]:


adata.obs[['leiden_scVI_0.7']].to_csv('../scanpy/231028_combine_thymoma_norm_scvi/leiden')


# In[173]:


adata


# In[175]:


adata.layers['scvi_normalized'].min()


# In[176]:


adata.layers['scvi_normalized'].max()


# In[177]:


adata.layers['counts'].max()


# In[179]:


df_cross = pd.crosstab(adata.obs['leiden_scVI_0.7'], adata.obs['sample_id'])
df_cross = df_cross / df_cross.sum()

plt.style.use({'axes.grid': False})
sns.clustermap(df_cross,figsize=(9,18), cmap='viridis')


# In[180]:


adata.obs[['MG_status', 'sample_id']].drop_duplicates()


# In[181]:


plt.rcParams['axes.grid'] = False
sc.pl.correlation_matrix(adata, 'leiden_scVI_0.7', figsize=(5,3.5))


# In[182]:


sc.pl.correlation_matrix(adata, 'clusters_orig', figsize=(8,7))


# In[218]:


adata.write(results_file)