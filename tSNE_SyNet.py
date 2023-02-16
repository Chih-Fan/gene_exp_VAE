import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

labeled_data = "./gene_exp_VAE/data/SyNet_Normalized_Expression_Par_Labeled_Data_Only.csv"
patient_data = "./gene_exp_VAE/data/Patient_information.csv"
gene_exp_df_labeled = pd.read_csv(labeled_data)
gene_exp_df_labeled_clean = pd.read_csv(labeled_data).iloc[: , 1:]
# print(gene_exp_df_labeled_clean.shape)

study_name_list = []
for name in gene_exp_df_labeled_clean.columns:
    study_name_list.append(name.split(';')[-1])
# print(len(study_name_list))

gene_exp_df_transposed = gene_exp_df_labeled_clean.T
# print(gene_exp_df_transposed.shape)
tsne = TSNE(n_components=2, random_state=0)
gene_exp_2d = tsne.fit_transform(gene_exp_df_transposed)

gene_exp_2d_df = pd.DataFrame({'tsne_1': gene_exp_2d[:,0], 'tsne_2': gene_exp_2d[:,1], 'label':study_name_list})
fig, ax = plt.subplots()
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=gene_exp_2d_df, ax=ax)
ax.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.0)
# plt.show()
plt.savefig("output.png", bbox_inches="tight")



