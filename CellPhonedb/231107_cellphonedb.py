#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce

#from vpolo.alevin import parser # to parse alevin output
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


# In[4]:


from cellphonedb.src.core.methods import cpdb_statistical_analysis_method


# In[78]:


version = '231107_cellphonedb'

import os
os.makedirs('../scanpy/{}'.format(version), exist_ok=True)

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, color_map='viridis', transparent=False, frameon=False)  # low dpi (dots per inch) yields small inline figures

sc.settings.figdir = '../scanpy/{}/graph'.format(version)


# ## prepare inputs

# In[27]:


adata = sc.read('../../MG_scRNAseq/scanpy/231031_combine_Xinetal/res_all_rmdoublet.h5ad')
adata


# In[45]:


df_meta = adata.obs[['cluster_L2']]
df_meta.columns = ['cell_type']
df_meta.head()


# In[46]:


df_meta.to_csv(f'../scanpy/{version}/meta.txt', sep='\t')


# In[36]:


adata = adata.raw.to_adata()


# In[38]:


adata.write(f'../scanpy/{version}/res_all_rmdoublet.raw.normlog1p.h5ad')


# In[8]:


df_loadings = pd.read_csv('../scanpy/231101_cell2loc/cell2location_map/CoLocatedComb/CoLocatedGroupsSklearnNMF_36625locations_50factors/cell_type_fractions_mean/n_fact8.csv', index_col=0)
df_loadings.columns = df_loadings.columns.str.replace('mean_cell_type_factorsfact_', 'fact_')
df_loadings.head()


# In[6]:


cutoff_loading = 0.1


# In[43]:


df_micro = df_loadings.melt(ignore_index=False)
df_micro = df_micro[df_micro['value'] >= cutoff_loading]
df_micro = df_micro.drop(['value'], axis=1).reset_index()
df_micro.columns = ['cell_type', 'microenvironment']
df_micro.head()


# In[53]:


dict_replace = {'CD4_Tph_CXCL13': 'CD4_Tfh_CXCL13',
               'CD8_Tph_CXCL13': 'CD8_Tem_CXCL13'}
df_micro['cell_type'] = [dict_replace[x] if x in dict_replace.keys() else x for x in df_micro['cell_type']]


# In[54]:


df_micro.to_csv(f'../scanpy/{version}/microenv.txt', sep='\t', index=False)


# ## Run cellphoneDB

# In[ ]:


cpdb_results = cpdb_statistical_analysis_method.call(
        cpdb_file_path = '../data/cellphonedb-data-4.1.0/cellphonedb.zip',
        meta_file_path = f'../scanpy/{version}/meta.txt',
        counts_file_path = f'../scanpy/{version}/res_all_rmdoublet.raw.normlog1p.h5ad',
        counts_data = 'hgnc_symbol',
        microenvs_file_path = f'../scanpy/{version}/microenv.txt',
        score_interactions = True,
        threshold = 0.1,
        output_path = f'../scanpy/{version}/results')


# ## Visualize

# In[2]:


import ktplotspy as kpy


# In[7]:


# read in the files
# 1) .h5ad file used for performing CellPhoneDB
adata = sc.read_h5ad(f'../scanpy/{version}/res_all_rmdoublet.raw.normlog1p.h5ad')

# 2) output from CellPhoneDB
means = pd.read_csv('../scanpy/231107_cellphonedb/results/statistical_analysis_means_11_07_2023_155610.txt', sep="\t")
pvals = pd.read_csv('../scanpy/231107_cellphonedb/results/statistical_analysis_pvalues_11_07_2023_155610.txt', sep="\t")
decon = pd.read_csv('../scanpy/231107_cellphonedb/results/statistical_analysis_deconvoluted_11_07_2023_155610.txt', sep="\t")


# In[9]:


kpy.plot_cpdb_heatmap(pvals=pvals, figsize=(15, 15), title="Sum of significant interactions")


# In[35]:


df_sigint = kpy.plot_cpdb_heatmap(pvals=pvals, return_tables=True)['count_network']


# In[21]:


df_sigint


# In[22]:


df_sigint.to_csv('../scanpy/231107_cellphonedb/results/significant_interactions.csv')


# In[37]:


df_sigint.values[np.triu_indices_from(df_sigint.values, k=1)] = np.nan


# In[39]:


df_sigint_melt = df_sigint.melt(ignore_index=False).reset_index().dropna()
df_sigint_melt.columns = ['source', 'target', 'interactions']


# In[41]:


df_sigint_melt.head()


# In[42]:


df_sigint_melt.to_csv('../scanpy/231107_cellphonedb/results/significant_interactions.tidy.csv')


# In[45]:


df_sigint.loc['nmTEC']


# In[46]:


df_sigint.loc['migDC']


# In[18]:


sc.pl.dotplot(adata, var_names=['NOTCH1', 'NOTCH2'], groupby='cluster_L2')


# In[50]:


list(adata.obs.cluster_L2.unique())


# In[74]:


list_show = [ 'cTEC',
'mTEC_CH3L1',
'mTEC_POU2F3',
'mTEC_AIRE',
'mTEC_GNB3',
'nmTEC',
'B_Naive',
'B_SM',
'B_GC',
'ASC',
'cDC1',
'cDC2',
'pDC',
'migDC',
'DN',
'DP',
'CyclingDNDP',
'CD4_Naive',
'Treg_Naive',
'Treg_Eff',
'CD4_Tcm_Th0',
'CD4_Tcm_Th0act',
'CD4_Tcm_Th17',
'CD4_Tfh',
'CD4_Tfh_CXCL13',
'CD4_Tem_Th1',
'CD4_Temra',]


# In[79]:


sc.pl.dotplot(adata[adata.obs.cluster_L2.isin(list_show)], var_names=[
                                                                      'CCL25', 'CCR9',
    'CCL19', 'CCR7',
                                                                      'CCL17', 'CCL22', 'CCR4', 
                                                                     'CXCL16', 'CXCR6',
                                                                      'CXCL12', 'CXCR4', 
                                                                      'CXCL10', 'CXCR3',
                                                                     'CXCL13', 'CXCR5'], 
              categories_order=list_show,
              groupby='cluster_L2', cmap='viridis', vmax=2.2,
             save='chemo.pdf')
