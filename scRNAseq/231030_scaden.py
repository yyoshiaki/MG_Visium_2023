#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

np.random.seed(100)
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = (1,1,1,1)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

pd.set_option('display.max_columns', 80)

import scipy.stats as stats
import statsmodels.stats.multitest as multi


# In[2]:


dir_img = 'scanpy/230713_combine_Xinetal/scaden/img'
dir_file = 'scanpy/230713_combine_Xinetal/scaden'
os.makedirs(dir_img, exist_ok=True)


# In[3]:


# df_scaden = pd.read_csv('scanpy/230713_combine_Xinetal/scaden/thymoma/pred.n5000.maskinf.step5000.txt', sep='\t', index_col=0)
# df_scaden = pd.read_csv('scanpy/230713_combine_Xinetal/scaden/thymoma/pred.n1000.step5000.txt', sep='\t', index_col=0)
# df_scaden = pd.read_csv('scanpy/230713_combine_Xinetal/scaden/thymoma/pred.n3000.maskinf.step5000.txt', sep='\t', index_col=0)
df_scaden = pd.read_csv('scanpy/230713_combine_Xinetal/scaden/thymoma/pred.n10000.step5000.txt', sep='\t', index_col=0)
df_scaden.head()


# In[4]:


sns.clustermap(df_scaden.T, z_score=0)


# In[5]:


df_cli = pd.read_csv('/home/yyasumizu/media32TB/bioinformatics/TCGA_thymoma/idep/THYM.clinical.xml.csv', index_col=0)
df_cli.index = df_cli.bcr_patient_barcode
df_cli = df_cli.dropna(subset=['primary_pathology_history_myasthenia_gravis'])
df_cli = df_cli[~df_cli.duplicated()]
df_cli.head()


# In[6]:


df_cli['days_to_birth'] = -df_cli['days_to_birth']


# In[7]:


dict_who = {'Thymoma; Type A' : 'A',
       'Thymoma; Type AThymoma; Type AB' : "A;AB", 
            'Thymoma; Type AB' : "AB", 'Thymoma; Type B1' : "B1", 
       'Thymoma; Type B1Thymoma; Type B2' : "B1;B2",
            'Thymoma; Type B2' : "B2",
       'Thymoma; Type B2Thymoma; Type B3' : "B2;B3", 
            'Thymoma; Type B3' : "B3", 
       'Thymoma; Type C':"C"}


# In[8]:


df_cli['WHO'] = [dict_who[x] for x in df_cli['primary_pathology_histological_type_list']]


# In[9]:


df_scaden.shape


# In[10]:


df_cli.shape


# In[11]:


barcodes = list(set(df_cli.index) & set(df_scaden.index))


# In[12]:


df_cli = df_cli.loc[barcodes]
df_scaden = df_scaden.loc[barcodes]


# In[13]:


df_merged = pd.merge(df_scaden, df_cli, left_index=True, right_index=True)


# In[14]:


df_merged


# In[15]:


sns.swarmplot(data=df_merged, x='primary_pathology_history_myasthenia_gravis', 
              y='nmTEC', color="white", edgecolor="gray", order=['YES', 'NO'])
sns.violinplot(data=df_merged, x='primary_pathology_history_myasthenia_gravis', 
               y='nmTEC', inner=None, order=['YES', 'NO'])


# In[16]:


list_p = []
list_f = []
for c in df_scaden.columns:
    plt.figure(figsize=(2,2))
#     sns.swarmplot(data=df_merged, x='primary_pathology_history_myasthenia_gravis', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_history_myasthenia_gravis', y=c, inner=None)
    sns.violinplot(data=df_merged, x='primary_pathology_history_myasthenia_gravis', y=c, order=['YES', 'NO'])
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title(c)
    
    s, p = stats.mannwhitneyu(df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] == "YES", c],
                  df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] == "NO", c])
    list_p.append(p)
    list_f.append(df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] == "YES", c].mean() / 
                 df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] == "NO", c].mean())


# In[17]:


plt.figure(figsize=(15,12))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(7,8,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
    sns.violinplot(data=df_merged, x='primary_pathology_history_myasthenia_gravis', y=c, order=['YES', 'NO'])
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('{}/violin_mg.pdf'.format(dir_img) , bbox_inches='tight')


# In[18]:


df_p = pd.DataFrame([list_f, list_p], columns=df_scaden.columns, index = ['fold', 'p']).T


# In[19]:


df_p['padj'] = multi.multipletests(list_p, method="fdr_bh", alpha=0.1)[1]


# In[20]:


df_p.sort_values(by='p')


# In[21]:


dict_who = {'Thymoma; Type A' : 'A',
       'Thymoma; Type AThymoma; Type AB' : "A;AB", 
            'Thymoma; Type AB' : "AB", 'Thymoma; Type B1' : "B1", 
       'Thymoma; Type B1Thymoma; Type B2' : "B1;B2",
            'Thymoma; Type B2' : "B2",
       'Thymoma; Type B2Thymoma; Type B3' : "B2;B3", 
            'Thymoma; Type B3' : "B3", 
       'Thymoma; Type C':"C"}


# In[22]:


list_who = ['A', 'AB', 'B1', 'B2', 'B3', 'C']


# In[23]:


sns.swarmplot(data=df_merged, x='WHO', y='nmTEC', order=list_who)
sns.swarmplot(data=df_merged, x='WHO', y='nmTEC', 
              color="white", edgecolor="gray", order=list_who)
sns.violinplot(data=df_merged, x='WHO', y='nmTEC', 
               inner=None, order=list_who)
plt.xticks(rotation=90)


# In[24]:


for c in df_scaden.columns:
    plt.figure(figsize=(3,2))
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
    sns.violinplot(data=df_merged, x='WHO', y=c, order=list_who)
    plt.ylabel(None)
    plt.title(c)
    plt.xticks(rotation=90)


# In[25]:


df_scaden.shape


# In[26]:


plt.figure(figsize=(15,9))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(7,8,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
    sns.violinplot(data=df_merged, x='WHO', y=c, order=list_who)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('{}/violin_who.pdf'.format(dir_img) , bbox_inches='tight')


# In[27]:


plt.figure(figsize=(15,15))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(8,7,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
    sns.violinplot(data=df_merged, x='WHO', hue='primary_pathology_history_myasthenia_gravis',
                   y=c, order=list_who, hue_order=['YES', 'NO'])
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
    plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig('{}/violin_mg_who.pdf'.format(dir_img) , bbox_inches='tight')


# In[28]:


plt.figure(figsize=(20,12))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(7,8,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
    sns.violinplot(data=df_merged, x='WHO', hue='primary_pathology_history_myasthenia_gravis', 
                   y=c, order=list_who, hue_order=['YES', 'NO'])
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
    plt.legend([],[], frameon=False)
plt.tight_layout()

plt.savefig('{}/violin_mg_who_wide.pdf'.format(dir_img) , bbox_inches='tight')


# In[29]:


sns.violinplot(data=df_merged,x='WHO' , hue='primary_pathology_history_myasthenia_gravis', 
               y='nmTEC', order=list_who, hue_order=['YES', 'NO'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('{}/violin_mg_who_nmTEC.pdf'.format(dir_img) , bbox_inches='tight')


# In[30]:


sns.violinplot(data=df_merged,x='WHO' , hue='primary_pathology_history_myasthenia_gravis', 
               y='Treg_Eff', order=list_who, hue_order=['YES', 'NO'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('{}/violin_mg_who_TregEff.pdf'.format(dir_img) , bbox_inches='tight')


# In[31]:


sns.violinplot(data=df_merged,x='WHO' , hue='primary_pathology_history_myasthenia_gravis', 
               y='B_GC', order=list_who, hue_order=['YES', 'NO'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('{}/violin_mg_who_GC_B.pdf'.format(dir_img) , bbox_inches='tight')


# In[32]:


sns.violinplot(data=df_merged,x='WHO' , hue='primary_pathology_history_myasthenia_gravis', 
               y='CD4_Tph_CXCL13', order=list_who, hue_order=['YES', 'NO'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('{}/violin_mg_who_CD4_Tph_CXCL13.pdf'.format(dir_img) , bbox_inches='tight')


# In[33]:


sns.violinplot(data=df_merged,x='WHO' , hue='primary_pathology_history_myasthenia_gravis', 
               y='migDC', order=list_who, hue_order=['YES', 'NO'])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig('{}/violin_mg_who_migDC.pdf'.format(dir_img) , bbox_inches='tight')


# In[34]:


palette = sns.color_palette()
list_color_mg = [palette[0] if x['primary_pathology_history_myasthenia_gravis'] =="YES" else palette[1] for _,x in df_merged.iterrows()]
dict_who_color = {'A' : palette[0],"A;AB":palette[7], 
            "AB":palette[1], "B1":palette[2], "B1;B2":palette[7],
            "B2":palette[3], "B2;B3":palette[7], 
            "B3" : palette[4], "C":palette[5]}
list_color_who = [dict_who_color[x] for x in df_merged['WHO']]

sns.clustermap(df_scaden.T, z_score=0, col_colors=[list_color_who, list_color_mg], figsize=(8,12))

plt.savefig('{}/clustermap_mg_who.pdf'.format(dir_img) , bbox_inches='tight')


# In[35]:


plt.scatter(df_merged['days_to_birth'], df_merged[c])


# In[36]:


plt.scatter(df_merged.loc[
    df_merged['primary_pathology_history_myasthenia_gravis'] =='YES', 'days_to_birth'], 
            df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] =='YES', c])
plt.scatter(df_merged.loc[
    df_merged['primary_pathology_history_myasthenia_gravis'] =='NO', 'days_to_birth'], 
            df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] =='NO', c])


# In[37]:


plt.figure(figsize=(20,18))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(7,8,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
#     sns.violinplot(data=df_merged, x='WHO', hue='primary_pathology_history_myasthenia_gravis', y=c, order=list_who)

    plt.scatter(df_merged.loc[
        df_merged['primary_pathology_history_myasthenia_gravis'] =='NO', 'days_to_birth'], 
                df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] =='NO', c],
               c='tab:orange')
    plt.scatter(df_merged.loc[
    df_merged['primary_pathology_history_myasthenia_gravis'] =='YES', 'days_to_birth'], 
            df_merged.loc[df_merged['primary_pathology_history_myasthenia_gravis'] =='YES', c],
            c='tab:blue')

    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
    plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig('{}/scatter_age_wide_mg.pdf'.format(dir_img) , bbox_inches='tight')


# In[38]:


df_merged['primary_pathology_histological_type_list'].unique()


# In[39]:


plt.figure(figsize=(20,18))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(7,8,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
#     sns.violinplot(data=df_merged, x='WHO', hue='primary_pathology_history_myasthenia_gravis', y=c, order=list_who)

    plt.scatter(df_merged.loc[
        df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type A', 'days_to_birth'], 
                df_merged.loc[df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type A', c])
    plt.scatter(df_merged.loc[
        df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type AB', 'days_to_birth'], 
                df_merged.loc[df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type AB', c])
    plt.scatter(df_merged.loc[
        df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B1', 'days_to_birth'], 
                df_merged.loc[df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B1', c])
    plt.scatter(df_merged.loc[
        df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B2', 'days_to_birth'], 
                df_merged.loc[df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B2', c])
    plt.scatter(df_merged.loc[
        df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B3', 'days_to_birth'], 
                df_merged.loc[df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B3', c])
    plt.scatter(df_merged.loc[
        df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type C', 'days_to_birth'], 
                df_merged.loc[df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type C', c])

    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
    plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig('{}/scatter_age_wide_who.pdf'.format(dir_img) , bbox_inches='tight')


# In[40]:


plt.figure(figsize=(24,24))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(9,6,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
#     sns.violinplot(data=df_merged, x='WHO', hue='primary_pathology_history_myasthenia_gravis', y=c, order=list_who)

    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type A') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type A') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), c],
               c='tab:blue', marker=".")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type AB') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type AB') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), c],
               c='tab:orange', marker=".")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B1') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B1') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), c],
               c='tab:green', marker=".")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B2') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B2') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), c],
               c='tab:red', marker=".")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B3') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B3') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), c],
               c='tab:purple', marker=".")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type C') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type C') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='NO'), c],
               c='tab:brown', marker=".")
    

    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type A') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type A') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), c],
               c='tab:blue', marker="x")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type AB') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type AB') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), c],
               c='tab:orange', marker="x")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B1') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B1') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), c],
               c='tab:green', marker="x")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B2') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B2') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), c],
               c='tab:red', marker="x")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B3') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type B3') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), c],
               c='tab:purple', marker="x")
    plt.scatter(df_merged.loc[
        (df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type C') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), 'days_to_birth'], 
             df_merged.loc[(df_merged['primary_pathology_histological_type_list'] =='Thymoma; Type C') &
        (df_merged['primary_pathology_history_myasthenia_gravis'] =='YES'), c],
               c='tab:brown', marker="x")
    
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
    plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig('{}/scatter_age_wide_who_mg.pdf'.format(dir_img) , bbox_inches='tight')


# In[ ]:





# In[41]:


plt.figure(figsize=(20,18))
for i,c in enumerate(df_scaden.columns):
    plt.subplot(7,8,i+1)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c)
#     sns.swarmplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, color="white", edgecolor="gray")
#     sns.violinplot(data=df_merged, x='primary_pathology_histological_type_list', y=c, inner=None)
#     sns.violinplot(data=df_merged, x='WHO', hue='primary_pathology_history_myasthenia_gravis', y=c, order=list_who)
    plt.scatter(df_merged['days_to_birth'], df_merged[c])
    plt.ylabel(None)
    plt.xlabel(None)
    plt.title(c)
    plt.xticks(rotation=90)
    plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig('{}/scatter_age_wide.pdf'.format(dir_img) , bbox_inches='tight')


# In[42]:


df_merged.plot.scatter(x='days_to_birth', y='primary_pathology_history_myasthenia_gravis')


# In[43]:


sns.violinplot(data=df_merged, x='primary_pathology_history_myasthenia_gravis',
               y='days_to_birth', order=['YES', 'NO'])


# In[44]:


sns.violinplot(data=df_merged, x='WHO', y='days_to_birth')


# In[ ]:





# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


y = [1 if x == "YES" else 0 for x in df_merged['primary_pathology_history_myasthenia_gravis']]
lr = LogisticRegression()
lr.fit(df_scaden, y)


# In[47]:


lr.predict_proba(df_scaden)[:,0]


# In[48]:


from sklearn import metrics


# In[49]:


fpr, tpr, thresholds = metrics.roc_curve(y, lr.predict_proba(df_scaden)[:,1])
metrics.auc(fpr, tpr)


# In[50]:


plt.figure(figsize = (5, 5)) #単一グラフの場合のサイズ比の与え方
plt.plot(fpr, tpr)
plt.xlabel('FPR: False Positive Rete', fontsize = 13)
plt.ylabel('TPR: True Positive Rete', fontsize = 13)
plt.grid(False)
plt.show()


# In[51]:


lr.coef_


# In[52]:


df_merged_conv = df_merged.copy()


# In[53]:


df_merged_conv = df_merged.copy()
df_merged_conv[df_scaden.columns] = \
    (df_merged_conv[df_scaden.columns] - df_merged_conv[df_scaden.columns].mean()) / df_merged_conv[df_scaden.columns].std(ddof=0)
df_merged_conv = df_merged_conv[df_merged_conv['WHO'].isin(list_who)]
df_merged_conv.columns = [x.replace('/', '_').replace(' ', '_').replace('(', '_').replace(')', '') for x in df_merged.columns]
df_merged_conv.to_csv('{}/merged.csv'.format(dir_file))


# In[54]:


from statsmodels.formula.api import ols


# In[55]:


model = ols("nmTEC ~ primary_pathology_history_myasthenia_gravis + WHO + days_to_birth + gender + 1", df_merged_conv).fit()
print(model.summary()) 


# In[56]:


model.pvalues


# In[57]:


model = ols("B_GC ~ primary_pathology_history_myasthenia_gravis + WHO + days_to_birth +gender + 1", df_merged_conv).fit()
print(model.summary()) 


# In[58]:


list_p = []
list_coef = []
for c in df_scaden.columns:
    model = ols("{} ~ primary_pathology_history_myasthenia_gravis + WHO + days_to_birth +gender + 1".format(
    c.replace('/', '_').replace(' ', '_').replace('(', '_').replace(')', '')), df_merged_conv).fit()
    list_p.append(model.pvalues['primary_pathology_history_myasthenia_gravis[T.YES]'])
    list_coef.append(model.params['primary_pathology_history_myasthenia_gravis[T.YES]'])


# In[59]:


df_p = pd.DataFrame([list_coef, list_p], columns=df_scaden.columns, index=['coef', 'p']).T
df_p['padj'] = multi.multipletests(list_p, method="fdr_bh", alpha=0.1)[1]
df_p = df_p.sort_values(by='p')
df_p['-logpadj'] = -np.log10(df_p['padj'])
df_p


# In[60]:


df_p['rank'] = df_p['-logpadj'] * df_p['coef']
df_p = df_p.sort_values(by='rank', ascending=False)
df_p['pos'] = range(df_p.shape[0])

df_p['m'] = 'o'
df_p.loc[df_p['padj'] < 0.05, 'm'] = 'D'


# In[61]:


import matplotlib.cm as cm
from matplotlib.colors import Normalize

cmap = cm.viridis
norm = Normalize(vmin=df_p['-logpadj'].min(), vmax=df_p['-logpadj'].min())


# In[62]:


plt.figure(figsize=(10,2))

# d_p = df_p[df_p['padj'] <= 0.05]
# plt.scatter(d_p['pos'], d_p['coef'], c=d_p['-logp'])

# d_p = df_p[df_p['padj'] > 0.05]
# plt.scatter(d_p['pos'], d_p['coef'], c=d_p['-logp'])

for pos, row in df_p.iterrows():
    plt.scatter(row['pos'], row['coef'], c=cmap(norm(row['-logpadj'])))
    
# plt.scatter(df_p['pos'], df_p['coef'], c=df_p['-logp'])

plt.axhline(y=0, xmin=0, xmax=df_p.shape[0], linestyle='--', c='grey')
plt.ylim(-0.8,0.8)
plt.xticks(range(df_p.shape[0]), df_p.index, rotation=90)
plt.colorbar()


# In[63]:


df_p['m'] = 'o'
df_p.loc[df_p['padj'] < 0.05, 'm'] = 'D'


# In[64]:


from adjustText import adjust_text

plt.figure(figsize=(8,8))
d_p = df_p[df_p['padj'] < 0.05]
plt.scatter(d_p['coef'], d_p['-logpadj'], color='r')

d_p = df_p[(df_p['padj'] > 0.05) & (df_p['padj'] < 0.2)]
plt.scatter(d_p['coef'], d_p['-logpadj'], color='orange')

d_p = df_p[df_p['padj'] > 0.2]
plt.scatter(d_p['coef'], d_p['-logpadj'], color='k')
plt.xlim(-1,1)
plt.xlabel('coefficient')
plt.ylabel('-logpadj')

texts = []
for pos,row in df_p.head(n=10).iterrows():
    texts.append(plt.text(row['coef'], row['-logpadj'], pos, size=9))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

plt.savefig('{}/volcano.pdf'.format(dir_img) , bbox_inches='tight')

