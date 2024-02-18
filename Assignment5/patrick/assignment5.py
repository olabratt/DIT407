import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import rand_score
from sklearn.metrics import accuracy_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools
import scipy.cluster.hierarchy as shc
# from umap import UMAP

columns = ["Area", "Permiter", "Compactness", "Kernel length", "Kernel width",
           "Asymmetry coefficient", "Groove length", "Class label"]
SEED = 0

df = pd.read_csv("Assignment5/seeds.tsv", sep='\t')
df.columns = columns
y = df["Class label"]

# Problem 1

scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(df.drop("Class label", axis=1)), columns = columns[:-1])


# Problem 2

inertia_values = []
k_values = [i for i in range(1, 10)]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=SEED).fit(X_normalized)
    inertia_values.append(kmeans.inertia_)

# plt.grid()
# plt.plot(k_values, inertia_values, 'o-')
# plt.show()

# Problem 3

colors = ["Red", "Green", "Blue"]
color_values = [colors[y[i] - 1] for i in range(y.size)]

# n_rows, n_columns = X_normalized.shape
# fig, axs = plt.subplots(n_columns - 1, n_columns - 1)
# for i, feature_1 in enumerate(X_normalized.columns):
#     for j, feature_2 in enumerate(X_normalized.columns[i + 1:]):
#         axs[i, j].scatter(X_normalized[feature_1], X_normalized[feature_2], c=color_values)
# plt.show()



# transformer = GaussianRandomProjection(2, random_state=SEED)
# X_normalized_transformed = transformer.fit_transform(X_normalized)
# plt.scatter(X_normalized_transformed[:,0], X_normalized_transformed[:,1], c=color_values)
# plt.show()

# reducer = UMAP(random_state=SEED)
# X_normalized_transformed = reducer.fit_transform(X_normalized)
# plt.scatter(X_normalized_transformed[:,0], X_normalized_transformed[:,1], c=color_values)
# plt.show()


# Problem 4

kmeans = KMeans(3, random_state=SEED).fit(X_normalized)
rand_index = rand_score(kmeans.labels_, y)
print("Rand index :", rand_index)

max_accuracy = 0
for perm in itertools.permutations(range(4)):
    accuracy = accuracy_score([perm[label] for label in kmeans.labels_], y)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_perm = perm

print("Best accuracy :", max_accuracy)
print("Best permutation :", dict(zip(range(4), best_perm)))

# Problem 5

clustering = AgglomerativeClustering(3).fit(X_normalized)
shc.dendrogram(shc.linkage(X_normalized, method='average'))
plt.show()