#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, color_map='viridis', transparent=False, frameon=False)  # low dpi (dots per inch) yields small inline figures

import matplotlib as mpl
# 2 lines below solved the facecolor problem.
# mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
sc.settings.autosave = True
sc.logging.print_header()

version = '231031_combine_Xinetal'

results_file = './scanpy/{}/res.h5ad'.format(version)
results_raw_file = './scanpy/{}/res.raw.h5ad'.format(version)
results_regressout_file = './scanpy/{}/res.regressout.h5ad'.format(version)
results_file_cellxgene = './scanpy/{}/res.cxg.h5ad'.format(version)

import os
os.makedirs('./scanpy/{}'.format(version), exist_ok=True)

sc.settings.figdir = './scanpy/{}/graph'.format(version)
sc.settings.cachedir = './scanpy/{}/cache'.format(version)
# %config InlineBackend.figure_format = 'retina'

import os
os.makedirs('./scanpy/{}'.format(version), exist_ok=True)
os.makedirs(sc.settings.figdir, exist_ok=True)


# ## preprocess : 230713_combine_Xinetal.ipynb

# ## Subcluseter L2,L3
# 
# - TEC: TEC-cell-labels-R7FYJ64X.csv
# - Fibro, VSMC: FibroblastVSMC-cell-labels-3SJU34DB.csv
# - T: matureT-cell-labels-STQFDNC3.csv
# - matT: matureT-cell-labels-K47LC3VV.csv
# - B: B-cell-labels-K43SYGTK.csv
# - Myelo: Myeloid-cell-labels-6I5ZEBLS.csv

# In[3]:


adata = sc.read('./scanpy/230713_combine_Xinetal/res.L1.h5ad')


# In[4]:


import psutil


def unused_portnumber(start=49152):
    # "LISTEN" 状態のポート番号をリスト化
    used_ports = [conn.laddr.port for conn in psutil.net_connections() if conn.status == 'LISTEN']
    for port in range(start, 65535 + 1):
        # 未使用のポート番号ならreturn
        if port not in set(used_ports):
            return port

            
def subcluster(adata, clu, cat='cluster_L1', n_top_genes=3000, n_neighbors=30, n_pcs=30, resolution=1, spread=1):
    adata = adata[adata.obs[cat] == clu].copy()
    adata_tmp = adata.copy()
    adata.X = adata.layers['counts'].copy()
    sc.pp.filter_genes(adata, min_cells=5)
    adata.X = adata_tmp[:,adata.var.index].X
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        subset=False,
        layer="counts",
        flavor="seurat_v3",
        batch_key="project"
    )
    sc.pl.highly_variable_genes(adata, log=True)
    
    sc.tl.pca(adata, svd_solver='arpack', use_highly_variable=True)
    sce.pp.harmony_integrate(adata, 'sample')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_pca_harmony')
    sc.tl.umap(adata, spread=spread)
    sc.tl.leiden(adata, key_added='leiden_{}'.format(clu), resolution=resolution)

    with plt.rc_context({"figure.figsize": (7, 7)}):
        sc.pl.umap(adata, color='leiden_{}'.format(clu), add_outline=True, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
               title=f'leiden_{cat}_{clu}_res_{resolution}', save=f'leiden_{cat}_{clu}_res_{resolution}')

    adata.uns['log1p']["base"] = None
    sc.tl.rank_genes_groups(adata, 'leiden_{}'.format(clu), method='wilcoxon')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save=f'{cat}_{clu}')
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, values_to_plot='logfoldchanges', 
                                    min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'{cat}_{clu}')

    adata.write(f'{sc.settings.figdir}/../{cat}_{clu}.h5ad')
    print(f'cd {sc.settings.figdir}/..')
    print(f'cellxgene launch {cat}_{clu}.h5ad --host 0.0.0.0 --port {unused_portnumber()} &')

    return adata


# ### TEC

# In[21]:


list_colors = ['black', 'turquoise', 'blue', 'yellow', 'green', 'red', 'grey']
sc.pl.umap(adata_TEC, color=['score_{}'.format(c) for c in list_colors], cmap='seismic',)


# In[5]:


adata_TEC = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_TEC.h5ad')
df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/TEC-cell-labels-R7FYJ64X.csv', comment='#', index_col=0)
adata_TEC.obs = pd.concat([adata_TEC.obs, df_annot], axis=1)
df_annot.head()


# In[6]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_TEC, color='cluster_L2_TEC_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_TEC_1st')


# In[11]:


sc.pl.umap(adata_TEC, color=['FOXN1', 'AIRE', 'KRT6A', 'KRT15', 'TF', 'CHRNA1', 'GABRA5', 'NEFM', 'NEFL',
                             'KRT17', 'KRT7', 'KRT18', 'SOX15', 'CCL19', 'CCL25', 'PSMB11', 'DSG1', 'CHAT', 'RBFOX3',
                             'PECAM1', 'LYVE1', 'ACTA2', 'PDPN', 'PDGFRA', 'CALB2'], s=20,
          save='TEC_markers.pdf')


# In[10]:


list_colors = ['black', 'turquoise', 'blue', 'yellow', 'green', 'red', 'grey']
sc.pl.umap(adata_TEC, color=['score_{}'.format(c) for c in list_colors], cmap='seismic', save='TEC_WGCNA.pdf')


# In[12]:


ax = sc.pl.correlation_matrix(adata_TEC, 'cluster_L2_TEC_1st', figsize=(5,3.5), save='TEC_corr.pdf')


# In[13]:


adata_TEC


# In[14]:


adata_TEC.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata_TEC, 'cluster_L2_TEC_1st', method='wilcoxon')
sc.pl.rank_genes_groups(adata_TEC, n_genes=25, sharey=False, save=f'cluster_L2_TEC_1st')
sc.pl.rank_genes_groups_dotplot(adata_TEC, n_genes=4, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_TEC_1st')


# In[30]:


sc.pl.rank_genes_groups_dotplot(adata_TEC, n_genes=6, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_TEC_1st_n6')

adata_TEC.write('./scanpy/{}/res.TEC.h5ad'.format(version))
# ## fibroblast

# In[16]:


adata_fibro = sc.read('./scanpy/230713_combine_Xinetal/res.fibro.h5ad')


# In[17]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_fibro, color='cluster_L2_fibroblastVSMC_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_fibroblastVSMC_1st')


# In[31]:


adata_fibro.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata_fibro, 'cluster_L2_fibroblastVSMC_1st', method='wilcoxon')
sc.pl.rank_genes_groups(adata_fibro, n_genes=25, sharey=False, save=f'cluster_L2_fibroblastVSMC_1st')
sc.pl.rank_genes_groups_dotplot(adata_fibro, n_genes=4, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_fibroblastVSMC_1st')


sc.pl.rank_genes_groups_dotplot(adata_fibro, n_genes=6, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_fibroblastVSMC_1st_n6',)

adata_fibro.write('./scanpy/{}/res.fibro.h5ad'.format(version))
# ## Immune cells

# ## T

# In[24]:


adata_matT = sc.read('./scanpy/230713_combine_Xinetal/res.T.h5ad')


# In[25]:


sc.pl.umap(adata_matT, color=['CD3E', 'CD4', 'CD8A', 'FCER1G', 'CCR7', 'FAS', 'CD28', 
                              'CXCR5', 'CXCL13', 'PDCD1', 'FOXP3', 'TBX21', 'CCR4', 'GATA3', 'RORC', 'CCR6',
                             'KLRB1', 'NR4A1', 'CXCR4', 'project', 'site'], save='matT_markers.pdf')


# In[26]:


sc.pl.dotplot(adata_matT, var_names=['CD3E', 'CD4', 'CD8A', 'FCER1G', 'CCR7', 'FAS', 'CD28', 
                              'CXCR5', 'CXCL13', 'PDCD1', 'FOXP3', 'TBX21', 'CCR4', 'GATA3', 'RORC', 'CCR6',
                             'KLRB1', 'NR4A1', 'CXCR4'], groupby='leiden_unassigned')


# In[27]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_matT[adata_matT.obs['project']=='Yasumizu_et_al'], color='minor_cluster', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='minor_cluster', save='minor_cluster_matT')


# In[26]:


adata_matT.var.loc[['CD8A', 'CD4']]


# In[28]:


df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/matureT-cell-labels-K47LC3VV.csv', comment='#', index_col=0)
# adata_matT.obs = pd.concat([adata_matT.obs, df_annot], axis=1)
df_annot.head()


# In[29]:


df_annot['cluster_L2_matT_1st'].unique()


# In[38]:


dict_edit = {'CD4_Tph_CXCL13': 'CD4_Tfh_CXCL13',
            'CD8_Tph_CXCL13': 'CD8_Tem_CXCL13'}


# In[43]:


adata_matT.obs['cluster_L2_matT_1st'] = [dict_edit[x] if x in dict_edit.keys() else x for x in adata_matT.obs['cluster_L2_matT_1st']]


# In[37]:


sc.pl.umap(adata_matT, color=['CD3E', 'CD4', 'CD8A', 'CCR7', 'FAS', 'CD28', 
                              'CXCR5', 'CXCL13', 'IL21', 'PDCD1', 'TIGIT', 'ICOS', 'BCL6', 'CCR2',
                              'FOXP3', 'TBX21', 'CCR4', 'GATA3', 'RORC', 'CCR6',
                             'KLRB1', 'NR4A1', 'CXCR4', 'project', 'site'], 
           s=15, save='matT_Tfh_markers.pdf')


# In[44]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_matT, color='cluster_L2_matT_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_matT_1st_v2')


# In[45]:


sc.pl.dotplot(adata_matT, var_names=['CD3E', 'CD4', 'CD8A', 'FCER1G', 'CCR7', 'FAS', 'CD28', 
                              'CXCR5', 'CXCL13', 'PDCD1', 'FOXP3', 'TBX21', 'CCR4', 'GATA3', 'RORC', 'CCR6',
                             'KLRB1', 'NR4A1', 'CXCR4'], groupby='cluster_L2_matT_1st')


# In[46]:


sc.tl.dendrogram(adata_matT, groupby='cluster_L2_matT_1st')


# In[47]:


adata_matT.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata_matT, 'cluster_L2_matT_1st', method='wilcoxon')
sc.pl.rank_genes_groups(adata_matT, n_genes=25, sharey=False, save=f'cluster_L2_matT_1st')
sc.pl.rank_genes_groups_dotplot(adata_matT, n_genes=4, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_matT_1st_v2')


sc.pl.rank_genes_groups_dotplot(adata_matT, n_genes=6, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_matT_1st_v2_n6',)


# In[48]:


adata_matT.write('./scanpy/{}/res.T.h5ad'.format(version))


# ## B

# In[50]:


adata_B = sc.read('./scanpy/230713_combine_Xinetal/res.B.h5ad')


# In[51]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_B, color='leiden_B', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='Leiden B', save='leiden_B')


# In[51]:


sc.pl.umap(adata_B, color=['PTPRC', 'CD3E', 'CD4', 'CST3', 'CD8A', 'CD19', 'CD79A', 'MS4A1', 'MEF2B', 'BCL6', 'CXCR5', 'CCR7', 'CD27', 'TBX21', 'ITGAX', 'MX1',
                          'CD38', 'SDC1', 'IGHM', 'IGHD', 'IGHA1', 'IGHA2', 'IGHG1', 'IGHE', 'KCNIP2', 'project', 'site'], s=20)


# In[244]:


sc.pl.umap(adata_B, color=['CD80', 'CD86', 'CD40', 'EBI3', 'CCR6'], s=20)


# In[236]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_B[adata_B.obs['project']=='Yasumizu_et_al'], color='minor_cluster', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='minor_cluster', save='minor_cluster_B')


# In[59]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_B, color='cluster_L2_B_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_B_1st')


# In[60]:


adata_B.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata_B, 'cluster_L2_B_1st', method='wilcoxon')
sc.pl.rank_genes_groups(adata_B, n_genes=25, sharey=False, save=f'cluster_L2_B_1st')
sc.pl.rank_genes_groups_dotplot(adata_B, n_genes=4, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_B_1st')


sc.pl.rank_genes_groups_dotplot(adata_B, n_genes=6, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_B_1st_n6',)

adata_B.write('./scanpy/{}/res.B.h5ad'.format(version))
# ## Myelo

# In[53]:


adata_Myelo = sc.read('./scanpy/230713_combine_Xinetal/res.Myeloid.h5ad')


# In[72]:


sc.pl.umap(adata_Myelo, color=['PTPRC', 'CD3E', 'CD4', 'CD8A', 'FOXP3', 'CCR7', 'IL21', 'LAMP3',
                          'FOS', 'CD28', 'CXCL13', 'CXCR5'], s=20)


# In[63]:


sc.pl.umap(adata_Myelo, color=['PTPRC', 'CD74', 'CCR7', 'LAMP3', 'FSCN1', 'CD274', 'site'], s=20, save='marker_migDC.pdf')


# In[56]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_Myelo, color='cluster_L2_Myeloid_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_Myeloid_1st')


# In[59]:


adata_Myelo.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata_Myelo, 'cluster_L2_Myeloid_1st', method='wilcoxon')
sc.pl.rank_genes_groups(adata_Myelo, n_genes=25, sharey=False, save=f'cluster_L2_Myeloid_1st')
sc.pl.rank_genes_groups_dotplot(adata_Myelo, n_genes=4, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_Myeloid_1st')


sc.pl.rank_genes_groups_dotplot(adata_Myelo, n_genes=6, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_Myeloid_1st_n6',)

adata_Myelo.write('./scanpy/{}/res.Myeloid.h5ad'.format(version))
# ## concat annotations

# In[64]:


adata = sc.read('./scanpy/230713_combine_Xinetal/res.L1.h5ad')


# In[65]:


df_tec = pd.read_csv('./scanpy/230713_combine_Xinetal/TEC-cell-labels-R7FYJ64X.csv', comment='#', index_col=0)
df_tec.columns = ['cluster_L2', 'cluster_L1']
df_tec.head()


# In[66]:


df_fibro = pd.read_csv('./scanpy/230713_combine_Xinetal/FibroblastVSMC-cell-labels-3SJU34DB.csv', comment='#', index_col=0)
df_fibro.columns = ['cluster_L2', 'cluster_L1']
df_fibro.head()


# In[67]:


df_t = pd.read_csv('./scanpy/230713_combine_Xinetal/matureT-cell-labels-STQFDNC3.csv', comment='#', index_col=0)
df_t.columns = ['cluster_L2', 'cluster_L1']
df_t = df_t[df_t['cluster_L1'] != 'unassigned']
df_t.head()


# In[68]:


df_matt = pd.read_csv('./scanpy/230713_combine_Xinetal/matureT-cell-labels-K47LC3VV.csv', comment='#', index_col=0)
df_matt.columns = ['cluster_L2', 'cluster_L1']
df_matt.head()


# In[70]:


dict_edit = {'CD4_Tph_CXCL13': 'CD4_Tfh_CXCL13',
            'CD8_Tph_CXCL13': 'CD8_Tem_CXCL13'}

df_matt.cluster_L2 = [dict_edit[x] if x in dict_edit.keys() else x for x in df_matt.cluster_L2]


# In[71]:


df_b = pd.read_csv('./scanpy/230713_combine_Xinetal/B-cell-labels-K43SYGTK.csv', comment='#', index_col=0)
df_b.columns = ['cluster_L2', 'cluster_L1']
df_b.head()


# In[72]:


df_myelo = pd.read_csv('./scanpy/230713_combine_Xinetal/Myeloid-cell-labels-6I5ZEBLS.csv', comment='#', index_col=0)
df_myelo.columns = ['cluster_L2', 'cluster_L1']
df_myelo.head()


# In[73]:


df_endo = adata.obs.loc[adata.obs.cluster_L1_all_1st == 'Endothelial', []]
df_endo['cluster_L2'] = 'Endothelial'
df_endo['cluster_L1'] = 'Endothelial'
df_endo.head()


# In[74]:


df = pd.concat([df_tec, df_fibro, df_endo, df_t, df_matt, df_b, df_myelo])


# In[75]:


adata.obs = pd.merge(adata.obs, df, left_index=True, right_index=True)


# In[76]:


adata.write(f'./scanpy/{version}/res.all.h5ad')


# In[77]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata, color='cluster_L2', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster_L2', save='cluster_L2')


# In[78]:


def subcluster_rmdoublet(adata, n_top_genes=3000, n_neighbors=30, n_pcs=30, spread=1):
    adata = adata[adata.obs['cluster_L1'] != 'Doublet'].copy()
    adata_tmp = adata.copy()
    adata.X = adata.layers['counts'].copy()
    sc.pp.filter_genes(adata, min_cells=5)
    adata.X = adata_tmp[:,adata.var.index].X
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        subset=False,
        layer="counts",
        flavor="seurat_v3",
        batch_key="project"
    )
    sc.pl.highly_variable_genes(adata, log=True)
    
    sc.tl.pca(adata, svd_solver='arpack', use_highly_variable=True)
    sce.pp.harmony_integrate(adata, 'sample')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X_pca_harmony')
    sc.tl.umap(adata, spread=spread)

    with plt.rc_context({"figure.figsize": (7, 7)}):
        sc.pl.umap(adata, color='cluster_L2', add_outline=True, legend_loc='on data',
               legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
               title=f'cluster_L2_rmdoublet', save=f'cluster_L2_rmdoublet')

    adata.write(f'{sc.settings.figdir}/../res_all_rmdoublet.h5ad')

    return adata


# In[79]:


adata_rmdoublet = subcluster_rmdoublet(adata, n_top_genes=3000, n_neighbors=45, n_pcs=None,spread=1)


# In[80]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_rmdoublet, color='cluster_L2', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster_L2', save='rmdoublet_cluster_L2')


# In[ ]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_rmdoublet, color='cluster_L2', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster_L2', save='rmdoublet_cluster_L2')


# In[ ]:





# In[ ]:





# # resume

# In[4]:


adata = sc.read(f'{sc.settings.figdir}/../res_all_rmdoublet.h5ad')


# In[84]:


list_colors = ['black', 'turquoise', 'blue', 'yellow', 'green', 'red', 'grey']
sc.pl.dotplot(adata, var_names=['score_{}'.format(c) for c in list_colors], groupby='cluster_L2',
         cmap='Blues', save='WGCNA_clusterL2.pdf')


# In[6]:


adata


# In[9]:


sc.pl.dotplot(adata, var_names=['HIVEP3', 'RGPD3', 'DNTT', 'PTCRA', 'AQP3', 'RAG1', 'RAG2', 'SMPD3', 'SATB1'],
             groupby='cluster_L2')


# In[ ]:





# In[ ]:





# ## scaden

# In[85]:


import os

dir_scaden = './scanpy/{}/scaden'.format(version)
os.makedirs('{}/thymoma'.format(dir_scaden), exist_ok=True)


# In[86]:


df_celltypes = pd.DataFrame(adata.obs['cluster_L2'])
df_celltypes.columns = ['Celltype']
df_celltypes = df_celltypes[df_celltypes['Celltype'] != 'Doublet']
df_celltypes.to_csv("{}/thymoma/thymoma_celltypes.txt".format(dir_scaden), sep="\t")
df_celltypes.head()


# In[87]:


adata[adata.obs['cluster_L2'] != 'Doublet']


# In[88]:


assert adata.shape[0] == adata.raw.shape[0]
df_count = pd.DataFrame(adata[adata.obs['cluster_L2'] != 'Doublet'].layers['counts'].todense(), 
                        columns = adata.raw.var.index)
df_count.to_csv('{}/thymoma/thymoma_counts.txt'.format(dir_scaden), sep='\t', index=False)
df_count.head()


# In[89]:


df_count.max()


# ```
# cd /home/yyasumizu/media32TB/bioinformatics/MG_scRNAseq/scanpy/230713_combine_Xinetal/scaden
# <!-- cp ~/media32TB/bioinformatics/TCGA_thymoma/201123_TCGATHYM_HTSeq_rawcounts.csv ./thymoma -->
# 
# # transfered to sol and executed in sol
# 
# cp ../../../210430_merged_thymoma_MG21.22.23.03/scaden/thymoma/TCGA-THYM.txt ./
# 
# docker run -v "$PWD:/data/" ghcr.io/kevinmenden/scaden/scaden-gpu:latest scaden simulate --data /data/thymoma -n 1000 --pattern "*_counts.txt" --prefix /data/thymoma/data.n1000
# docker run -v "$PWD:/data/" ghcr.io/kevinmenden/scaden/scaden-gpu:latest scaden process /data/thymoma/data.n1000.h5ad /data/thymoma/TCGA-THYM.txt --processed_path /data/thymoma/process.n1000.h5ad
# docker run -v "$PWD:/data/" ghcr.io/kevinmenden/scaden/scaden-gpu:latest scaden train /data/thymoma/process.n1000.h5ad --steps 5000 --model_dir /data/thymoma/model.n1000
# docker run -v "$PWD:/data/" ghcr.io/kevinmenden/scaden/scaden-gpu:latest scaden predict --model_dir /data/thymoma/model.n1000 /data/thymoma/TCGA-THYM.txt --outname /data/thymoma/pred.n1000.step5000.txt
# ```

# In[ ]:




