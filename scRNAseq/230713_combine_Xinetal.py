#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=80, color_map='viridis', transparent=False, frameon=False)  # low dpi (dots per inch) yields small inline figures

import matplotlib as mpl
# 2 lines below solved the facecolor problem.
# mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
sc.settings.autosave = True
sc.logging.print_header()

version = '230713_combine_Xinetal'

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


# In[ ]:


adata_yy = sc.read('./scanpy/210711_merged_thymoma_MG21.22.23.03/merge_cg.minor_cluster.h5ad')
adata_yy_raw = sc.read('./scanpy/210711_merged_thymoma_MG21.22.23.03/merge.raw.h5ad')
adata_yy.X = adata_yy_raw[adata_yy.obs.index].X
adata_yy.obs.index = 'Yasumizu-' + adata_yy.obs.index
adata_yy.obs['project'] = 'Yasumizu_et_al'
adata_yy


# In[4]:


list_a = []
for i in range(1,7):
    print(i)
    a = sc.read_10x_h5(f'./data/PRJCA009311_thymoma_scRNAseq/T0{i}.filtered_feature_bc_matrix.h5')
    a.var_names_make_unique()
    a.obs['sample'] = f'T0{i}'
    a.obs['site'] = 'Thymus'
    
    # apply scrublet
    sce.pp.scrublet(a)
    sce.pl.scrublet_score_distribution(a)
    a = a[~a.obs.predicted_doublet]
    list_a.append(a)
adata_xin = sc.concat(list_a)
adata_xin.obs['project'] = 'Xin_et_al'
adata_xin.obs['major_cluster'] = 'null'
adata_xin.obs['minor_cluster'] = 'null'
adata_xin.obs.index = 'Xin-' + adata_xin.obs.index


# In[5]:


a.X.max()


# In[6]:


adata_xin


# In[7]:


sc.pp.filter_cells(adata_xin, min_genes=200)
adata_xin.var['mt'] = adata_xin.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata_xin, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[8]:


sc.pl.violin(adata_xin, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)


# In[9]:


adata_xin = adata_xin[adata_xin.obs.n_genes_by_counts < 8000, :]
adata_xin = adata_xin[adata_xin.obs.pct_counts_mt < 20, :]


# In[10]:


adata = sc.concat([adata_yy, adata_xin])
adata.obs_names_make_unique()
adata.var['mt'] = adata.var_names.str.startswith('MT-')
# adata = adata[:,~adata.var.mt]
adata = adata[:,(~ adata.var.index.str.startswith('IGKV')) & 
              (~ adata.var.index.str.startswith('IGLV')) &
             (~ adata.var.index.str.startswith('IGHV')) &
             (~ adata.var.index.str.startswith('IGLC')) &
             (~ adata.var.index.str.startswith('TRAV')) &
             (~ adata.var.index.str.startswith('TRBV')) &
             (~ adata.var.index.str.startswith('TRAJ')) &
             (~ adata.var.index.str.startswith('TRBD')) &
             (~ adata.var.index.str.startswith('TRBJ')) &
             (~ adata.var.index.str.startswith('TRG')) &
             (~ adata.var.index.str.startswith('TRD'))].copy()

print(adata)

adata.layers['counts'] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()


# In[11]:


sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    subset=False,
    layer="counts",
    flavor="seurat_v3",
    batch_key="project"
)

sc.pl.highly_variable_genes(adata, log=True)


# In[12]:


sc.pp.scale(adata, max_value=10)
cell_cycle_genes = [x.strip() for x in 
                    open('./data/regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes
                    if x in adata.var_names]
sc.tl.score_genes_cell_cycle(adata, 
                             s_genes=s_genes, g2m_genes=g2m_genes)


# In[13]:


sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt', 'S_score', 'G2M_score'])
sc.pp.scale(adata, max_value=10)


# In[14]:


adata.write(results_regressout_file)


# In[15]:


adata = sc.read(results_regressout_file)


# In[16]:


sc.tl.pca(adata, svd_solver='arpack', use_highly_variable=True)
sce.pp.harmony_integrate(adata, 'sample')
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50, use_rep='X_pca_harmony')
sc.tl.umap(adata, spread=2)
sc.tl.leiden(adata)


# In[17]:


sc.pl.umap(adata, color=['PTPRC', 'CD3E', 'CD4', 'CD8A', 'CCR7', 'FOXP3',
                         'CXCL13', 'CXCL12', 'CD79A', 'EPCAM', 'FOXN1', 'ITGAX',
                         'KRT19', 'PECAM1', 'FN1', 'PDPN', 'PDGFRA', 'RBFOX3', 'ACTA2', 'MKI67'], s=5)


# In[21]:


sc.pl.dotplot(adata, var_names=['PTPRC', 'CD3E', 'CD4', 'CD8A', 'CCR7', 'FOXP3',
                         'CXCL13', 'CD79A', 'EPCAM', 'FOXN1', 'ITGAX',
                         'KRT19', 'PECAM1', 'FN1', 'PDPN', 'PDGFRA', 'RBFOX3', 'ACTA2'], 
              groupby='leiden'
             )


# In[20]:


sc.pl.umap(adata, color=['AIRE', 'KRT6A', 'CHRNA1', 'GABRA5', 'NEFL', 'KRT17', 'PSMB11'], s=20)


# In[10]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata, color='leiden', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='Leiden', save=False)


# In[22]:


adata.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
sc.pl.rank_genes_groups_dotplot(adata, n_genes=4, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr')

sc.pl.umap(adata, color=['sample', 'project', 'major_cluster', 'minor_cluster'])
# In[23]:


sc.pl.umap(adata, color=['sample', 'project', 'site'])


# In[24]:


adata.write('./scanpy/{}/res.all.h5ad'.format(version))


# In[ ]:




dict_cluster_L1 = {
    "0" : "T",
    "1" : "T",
    "2" : "B",
    "3" : "T",
    "4" : "Myelo",
    "5" : "T",
    "6" : "T",
    "7" : "T",
    "8" : "T",
    "9" : "T",
    "10" : "B",
    "11" : "T",
    "12" : "nonImmune",
    "13" : "T",
    "14" : "T",
    "15" : "nonImmune",
    "16" : "T",
    "17" : "T",
    "18" : "T",
    "19" : "B",
    "20" : "nonImmune",
    "21" : "nonImmune",
    "22" : "Myelo",
    "23" : "Myelo",
    "24" : 'B'
}

adata.obs['cluster_L1'] = [dict_cluster_L1[x] for x in adata.obs['leiden']]
adata.obs['cluster_L1'] = adata.obs['cluster_L1'].astype('category')
# In[30]:


df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/all-cell-labels-66EGP6SU.csv', comment='#', index_col=0)
adata.obs['cluster_L1_all_1st'] = df_annot['cluster_L1_1st']


# In[17]:


with plt.rc_context({"figure.figsize": (4, 4)}):
    sc.pl.umap(adata, color='cluster_L1_all_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=5,
           title='Cluster L1', save='clusterL1.pdf')


# In[32]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata[adata.obs['project']=='Yasumizu_et_al'], color='minor_cluster', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='minor_cluster', save='minor_cluster')


# In[33]:


list_colors = ['black', 'turquoise', 'blue', 'yellow', 'green', 'red', 'grey']
list_gene_colors = []
for c in list_colors:
    d = pd.read_csv('../TCGA_thymoma/200703_DSeq2WGCNA/{}.txt'.format(c), header=None)
    list_gene_colors.append(list(d[0]))

for c,g in zip(list_colors, list_gene_colors):
    sc.tl.score_genes(adata, g, score_name='score_{}'.format(c), use_raw=True)


# In[34]:


sc.pl.umap(adata, color=['score_{}'.format(c) for c in list_colors], cmap='seismic',)


# In[35]:


adata.write('./scanpy/{}/res.L1.h5ad'.format(version))


# ## Subcluseter L2,L3
# 
# - TEC: TEC-cell-labels-R7FYJ64X.csv
# - Fibro, VSMC: FibroblastVSMC-cell-labels-3SJU34DB.csv
# - T: matureT-cell-labels-STQFDNC3.csv
# - matT: matureT-cell-labels-K47LC3VV.csv
# - B: B-cell-labels-K43SYGTK.csv
# - Myelo: Myeloid-cell-labels-6I5ZEBLS.csv

# In[61]:


adata = sc.read('./scanpy/{}/res.L1.h5ad'.format(version))


# In[8]:


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

# In[16]:


adata_TEC = subcluster(adata, 'TEC', cat="cluster_L1_all_1st", n_top_genes=3000, resolution=1)


# In[24]:


adata_TEC = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_TEC.h5ad')


# In[17]:


cat = 'cluster_L1'
clu = 'TEC'
resolution = 1
with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_TEC, color='leiden_{}'.format(clu), add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title=f'leiden_{cat}_{clu}_res_{resolution}', save=f'leiden_{cat}_{clu}_res_{resolution}')


# In[18]:


sc.pl.umap(adata_TEC, color=['FOXN1', 'AIRE', 'KRT6A', 'KRT15', 'TF', 'CHRNA1', 'GABRA5', 'NEFM', 'NEFL',
                             'KRT17', 'KRT7', 'KRT18', 'SOX15', 'CCL19', 'CCL25', 'PSMB11', 'DSG1', 'CHAT', 'RBFOX3',
                             'PECAM1', 'LYVE1', 'ACTA2', 'PDPN', 'PDGFRA', 'CALB2'], s=20)


# In[25]:


sc.pl.umap(adata_TEC, color=['sample', 'project', 'site'])


# In[26]:


sc.pl.umap(adata_TEC, color=['FEZF2', 'CD80', 'HLA-DRB1', 'KRT1', 'KRT14', 'DCLK1', 'DLL4', 'KRT10', 'n_genes_by_counts'], s=20)


# In[27]:


sc.pl.umap(adata_TEC, color=['PSMB11', 'CLDN4', 'CCL21', 'SPIB', 'AIRE',
                                   'FEZF2', 'IVL', 'NEUROD1', 'MYOD1', 'MPZ', 'MKI67'], s=20)


# In[183]:


sc.pl.umap(adata_TEC, color=['PTPRC', 'CD4', 'CD8A', 'PLA2G2A', 'DPP4', 'SMPD3',
                                   'SEMA3C', 'MSLN', 'SPON2', 'CXCL13', 'DES', 'ENPP2', 'POSTN', 
                                  'MSLN', 'UPK3B', 'PRG4'], s=20)


# In[28]:


sc.pl.umap(adata_TEC, color=['KRT14', 'CHI3L1', 'KRT19', 'KRT5', 'ASCL1', 'POU2F3', 'POU3F1', 'GNB3', 'S100A14'], s=20)


# In[21]:


list_colors = ['black', 'turquoise', 'blue', 'yellow', 'green', 'red', 'grey']
sc.pl.umap(adata_TEC, color=['score_{}'.format(c) for c in list_colors], cmap='seismic',)

# dict_cluster_L2 = {
#     "0" : "mTEC FEZF2",
#     "1" : "Doublet",
#     "2" : "mTEC KRT14",
#     "3" : "Endothelial",
#     "4" : "Fibroblast",
#     "5" : "mTEC S100A14",
#     "6" : "mTEC CHI3L1",
#     "7" : "mTEC POU2F3",
#     "8" : "mTEC FEZF2",
#     "9" : "mTEC CHI3L1",
#     "10" : "cTEC",
#     "11" : "VSMC",
#     "12" : "mTEC AIRE",
#     "13" : "nmTEC",
#     "14" : "Mesothelial"
# }

dict_cluster_L2 = {
    "0" : "mTEC FEZF2",
    "1" : "Doublet",
    "2" : "mTEC KRT14",
    "3" : "Endothelial",
    "4" : "Fibroblast",
    "5" : "mTEC S100A14",
    "6" : "mTEC CHI3L1",
    "7" : "mTEC POU2F3",
    "8" : "mTEC FEZF2",
    "9" : "nmTEC",
    "10" : "cTEC",
    "11" : "VSMC",
    "12" : "mTEC AIRE",
    "13" : "nmTEC",
    "14" : "Mesothelial"
}

adata_nonimmune.obs['cluster_L2'] = [dict_cluster_L2[x] for x in adata_nonimmune.obs['leiden_nonImmune']]
adata_nonimmune.obs['cluster_L2'] = adata_nonimmune.obs['cluster_L2'].astype('category')
# In[20]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_TEC[adata_TEC.obs['project']=='Yasumizu_et_al'], color='minor_cluster', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='minor_cluster', save='minor_cluster_TEC')


# In[4]:


adata_TEC = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_TEC.h5ad')
df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/TEC-cell-labels-R7FYJ64X.csv', comment='#', index_col=0)
adata_TEC.obs = pd.concat([adata_TEC.obs, df_annot], axis=1)
df_annot.head()


# In[35]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_TEC, color='cluster_L2_TEC_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_TEC_1st')


# In[37]:


ax = sc.pl.correlation_matrix(adata_TEC, 'cluster_L2_TEC_1st', figsize=(5,3.5))


# In[5]:


adata_TEC


# In[7]:


adata_TEC.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata_TEC, 'cluster_L2_TEC_1st', method='wilcoxon')
sc.pl.rank_genes_groups(adata_TEC, n_genes=25, sharey=False, save=f'cluster_L2_TEC_1st')
sc.pl.rank_genes_groups_dotplot(adata_TEC, n_genes=4, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_TEC_1st')


# In[9]:


sc.pl.rank_genes_groups_dotplot(adata_TEC, n_genes=6, values_to_plot='logfoldchanges', 
                                min_logfoldchange=1, vmax=7, vmin=-7, cmap='bwr', save=f'cluster_L2_TEC_1st')


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


adata_TEC.write('./scanpy/{}/res.TEC.h5ad'.format(version))


# ## fibroblast

# In[185]:


adata_fibro = subcluster(adata, 'Fibroblast_VSMC', cat="cluster_L1_all_1st", n_top_genes=1000, resolution=1)


# In[77]:


adata_fibro = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_Fibroblast_VSMC.h5ad')
df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/FibroblastVSMC-cell-labels-3SJU34DB.csv', comment='#', index_col=0)
adata_fibro.obs = pd.concat([adata_fibro.obs, df_annot], axis=1)
df_annot.head()


# In[79]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_fibro, color='cluster_L2_fibroblastVSMC_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_fibroblastVSMC_1st')


# In[78]:


adata_fibro.write('./scanpy/{}/res.fibro.h5ad'.format(version))


# ## Immune cells

# ## T

# In[9]:


# adata_T = subcluster(adata, 'T', n_top_genes=3000, resolution=1.5)
adata_T = subcluster(adata, 'T', cat="cluster_L1_all_1st", n_top_genes=3000, n_neighbors=15, n_pcs=25, resolution=2)
# adata_T = subcluster(adata, 'T', cat='cluster_L1', n_top_genes=3000, n_neighbors=5, n_pcs=30, resolution=2.5)


# In[10]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_T, color='leiden_T', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='Leiden T', save='leiden_T')


# In[ ]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_T[adata_T.obs['project']=='Yasumizu_et_al'], color='minor_cluster', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='minor_cluster', save='minor_cluster_T')


# In[ ]:


adata_T


# In[24]:


sc.pl.dotplot(adata_T, var_names=['CD3E', 'CD4', 'CD8A', 'MKI67', 'MS4A1', 'CST3', 'EPCAM'], groupby='leiden_T')


# In[22]:


sc.pl.umap(adata_T, color=['PTPRC', 'CD3E', 'CD4', 'CD8A', 'MKI67', 'FOXP3', 'CCR7',
                          'FOS', 'CD28', 'CXCL13', 'CXCR5', 'CXCR4', 'NKG7', 'NCAM1'], s=3)


# In[34]:


sc.pl.umap(adata_T, color=['PTPRC', 'CD3E', 'CD4', 'CD8A', 'FOXP3', 'CCR7',
                          'FOS', 'CD28', 'CXCL13', 'CXCR5', 'CXCR4', 'NKG7', 'NCAM1'], s=3)


# In[201]:


sc.pl.umap(adata_T, color=['FTH1'], s=3)


# In[195]:


sc.pl.umap(adata_T, color=['sample', 'project', 'site'])


# In[70]:


adata_T = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_T.h5ad')
df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/matureT-cell-labels-STQFDNC3.csv', comment='#', index_col=0)
adata_T.obs = pd.merge(adata_T.obs, df_annot, how='left', left_index=True, right_index=True)
df_annot.head()


# In[6]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_T[adata_T.obs['project']=='Yasumizu_et_al'], color='cluster_L2_T_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster_L2_T_1st', save='cluster_L2_T_1st')


# In[19]:


adata_matT = subcluster(adata_T, 'unassigned', cat='cluster_L2_T_1st', n_top_genes=3000, n_neighbors=15, n_pcs=10, resolution=2.5)


# In[20]:


adata_matT


# In[21]:


sc.pl.umap(adata_matT, color=['CD3E', 'CD4', 'CD8A', 'FCER1G', 'CCR7', 'FAS', 'CD28', 
                              'CXCR5', 'CXCL13', 'PDCD1', 'FOXP3', 'TBX21', 'CCR4', 'GATA3', 'RORC', 'CCR6',
                             'KLRB1', 'NR4A1', 'CXCR4', 'project', 'site'])


# In[40]:


sc.pl.dotplot(adata_matT, var_names=['CD3E', 'CD4', 'CD8A', 'FCER1G', 'CCR7', 'FAS', 'CD28', 
                              'CXCR5', 'CXCL13', 'PDCD1', 'FOXP3', 'TBX21', 'CCR4', 'GATA3', 'RORC', 'CCR6',
                             'KLRB1', 'NR4A1', 'CXCR4'], groupby='leiden_unassigned')


# In[29]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_matT[adata_matT.obs['project']=='Yasumizu_et_al'], color='minor_cluster', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='minor_cluster', save='minor_cluster_matT')


# In[26]:


adata_matT.var.loc[['CD8A', 'CD4']]


# In[63]:


adata_matT = sc.read('./scanpy/230713_combine_Xinetal/cluster_L2_T_1st_unassigned.h5ad')
df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/matureT-cell-labels-K47LC3VV.csv', comment='#', index_col=0)
adata_matT.obs = pd.concat([adata_matT.obs, df_annot], axis=1)
df_annot.head()


# In[64]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_matT, color='cluster_L2_matT_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_matT_1st')


# In[45]:


sc.pl.dotplot(adata_matT, var_names=['CD3E', 'CD4', 'CD8A', 'FCER1G', 'CCR7', 'FAS', 'CD28', 
                              'CXCR5', 'CXCL13', 'PDCD1', 'FOXP3', 'TBX21', 'CCR4', 'GATA3', 'RORC', 'CCR6',
                             'KLRB1', 'NR4A1', 'CXCR4'], groupby='cluster_L2_matT_1st')


# In[65]:


adata_matT.write('./scanpy/{}/res.T.h5ad'.format(version))


# ## B

# In[192]:


adata_B = subcluster(adata, 'B', cat="cluster_L1_all_1st", n_top_genes=3000, n_neighbors=30, n_pcs=10, resolution=0.7, spread=1)


# In[50]:


adata_B = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_B.h5ad')


# In[138]:


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


# In[58]:


adata_B = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_B.h5ad')
df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/B-cell-labels-K43SYGTK.csv', comment='#', index_col=0)
adata_B.obs = pd.concat([adata_B.obs, df_annot], axis=1)
df_annot.head()


# In[59]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_B, color='cluster_L2_B_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_B_1st')


# In[60]:


adata_B.write('./scanpy/{}/res.B.h5ad'.format(version))


# ## Myelo

# In[68]:


adata.obs["cluster_L1_all_1st"]


# In[ ]:





# In[69]:


# adata_Myelo = subcluster(adata, 'Myelo' )
adata_Myelo = subcluster(adata, 'Myeloid', cat="cluster_L1_all_1st", n_top_genes=3000, resolution=1)


# In[73]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_Myelo, color='leiden_Myeloid', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='Leiden Myelo', save='leiden_Myeloid')


# In[72]:


sc.pl.umap(adata_Myelo, color=['PTPRC', 'CD3E', 'CD4', 'CD8A', 'FOXP3', 'CCR7', 'IL21', 'LAMP3',
                          'FOS', 'CD28', 'CXCL13', 'CXCR5'], s=20)


# In[101]:


adata_Myelo = sc.read('./scanpy/230713_combine_Xinetal/cluster_L1_all_1st_Myeloid.h5ad')
df_annot = pd.read_csv('./scanpy/230713_combine_Xinetal/Myeloid-cell-labels-6I5ZEBLS.csv', comment='#', index_col=0)
adata_Myelo.obs = pd.concat([adata_Myelo.obs, df_annot], axis=1)
df_annot.head()


# In[102]:


with plt.rc_context({"figure.figsize": (7, 7)}):
    sc.pl.umap(adata_Myelo, color='cluster_L2_Myeloid_1st', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster L2', save='cluster_L2_Myeloid_1st')


# In[103]:


adata_Myelo.write('./scanpy/{}/res.Myeloid.h5ad'.format(version))


# ## concat annotations

# In[3]:


adata = sc.read('./scanpy/{}/res.L1.h5ad'.format(version))


# In[4]:


df_tec = pd.read_csv('./scanpy/230713_combine_Xinetal/TEC-cell-labels-R7FYJ64X.csv', comment='#', index_col=0)
df_tec.columns = ['cluster_L2', 'cluster_L1']
df_tec.head()


# In[5]:


df_fibro = pd.read_csv('./scanpy/230713_combine_Xinetal/FibroblastVSMC-cell-labels-3SJU34DB.csv', comment='#', index_col=0)
df_fibro.columns = ['cluster_L2', 'cluster_L1']
df_fibro.head()


# In[6]:


df_t = pd.read_csv('./scanpy/230713_combine_Xinetal/matureT-cell-labels-STQFDNC3.csv', comment='#', index_col=0)
df_t.columns = ['cluster_L2', 'cluster_L1']
df_t = df_t[df_t['cluster_L1'] != 'unassigned']
df_t.head()


# In[7]:


df_matt = pd.read_csv('./scanpy/230713_combine_Xinetal/matureT-cell-labels-K47LC3VV.csv', comment='#', index_col=0)
df_matt.columns = ['cluster_L2', 'cluster_L1']
df_matt.head()


# In[8]:


df_b = pd.read_csv('./scanpy/230713_combine_Xinetal/B-cell-labels-K43SYGTK.csv', comment='#', index_col=0)
df_b.columns = ['cluster_L2', 'cluster_L1']
df_b.head()


# In[9]:


df_myelo = pd.read_csv('./scanpy/230713_combine_Xinetal/Myeloid-cell-labels-6I5ZEBLS.csv', comment='#', index_col=0)
df_myelo.columns = ['cluster_L2', 'cluster_L1']
df_myelo.head()


# In[10]:


df_endo = adata.obs.loc[adata.obs.cluster_L1_all_1st == 'Endothelial', []]
df_endo['cluster_L2'] = 'Endothelial'
df_endo['cluster_L1'] = 'Endothelial'
df_endo.head()


# In[11]:


df = pd.concat([df_tec, df_fibro, df_endo, df_t, df_matt, df_b, df_myelo])


# In[12]:


adata.obs = pd.merge(adata.obs, df, left_index=True, right_index=True)


# In[13]:


adata.write('./scanpy/230713_combine_Xinetal/res.all.h5ad')


# In[14]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata, color='cluster_L2', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster_L2', save='cluster_L2')


# In[19]:


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


# In[ ]:


adata_rmdoublet = subcluster_rmdoublet(adata, n_top_genes=3000, n_neighbors=45, n_pcs=None,spread=1)


# In[25]:


with plt.rc_context({"figure.figsize": (17, 17)}):
    sc.pl.umap(adata_rmdoublet, color='cluster_L2', add_outline=True, legend_loc='on data',
           legend_fontsize=12, legend_fontoutline=2,frameon=False, s=30,
           title='cluster_L2', save='rmdoublet_cluster_L2')


# # resume

# In[3]:


adata = sc.read('./scanpy/230713_combine_Xinetal/res.all.h5ad')


# In[12]:


list_colors = ['black', 'turquoise', 'blue', 'yellow', 'green', 'red', 'grey']
sc.pl.dotplot(adata, var_names=['score_{}'.format(c) for c in list_colors], groupby='cluster_L2')


# In[6]:


adata


# ## scaden

# In[5]:


import os

dir_scaden = './scanpy/{}/scaden'.format(version)
os.makedirs('{}/thymoma'.format(dir_scaden), exist_ok=True)


# In[11]:


df_celltypes = pd.DataFrame(adata.obs['cluster_L2'])
df_celltypes.columns = ['Celltype']
df_celltypes = df_celltypes[df_celltypes['Celltype'] != 'Doublet']
df_celltypes.to_csv("{}/thymoma/thymoma_celltypes.txt".format(dir_scaden), sep="\t")
df_celltypes.head()


# In[12]:


adata[adata.obs['cluster_L2'] != 'Doublet']


# In[13]:


assert adata.shape[0] == adata.raw.shape[0]
df_count = pd.DataFrame(adata[adata.obs['cluster_L2'] != 'Doublet'].layers['counts'].todense(), 
                        columns = adata.raw.var.index)
df_count.to_csv('{}/thymoma/thymoma_counts.txt'.format(dir_scaden), sep='\t', index=False)
df_count.head()


# In[14]:


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




