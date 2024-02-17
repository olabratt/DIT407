import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

columns = ["Area", "Permiter", "Compactness", "Kernel length", "Kernel width",
           "Asymmetry coefficient", "Groove length", "Class label"]
SEED = 0

df = pd.read_csv("Assignment5/seeds.tsv", sep='\t')
df.columns = columns
y = df["Class label"]

# Problem 1

scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(df.drop("Class label", axis=1)), columns = columns[:-1])

X_train, y_train, X_test, y_test = train_test_split(X_normalized, y, test_size = 0.2, random_state=SEED)

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

n_rows, n_columns = X_normalized.shape
fig, axs = plt.subplots(int(n_columns / 2) + 1, n_columns - 1)
for i, feature_1 in enumerate(X_normalized.columns):
    for j, feature_2 in enumerate(X_normalized.columns[i + 1:]):
        offset = int((i + 1) * (i + 2) / 2)
        offset_i = int(offset / n_columns)
        offset_j = offset - offset_i * n_columns
        axs[i - offset_i, j - offset_j].scatter(X_normalized[feature_1], X_normalized[feature_2], c=color_values)

plt.show()


transformer = GaussianRandomProjection(2, random_state=SEED)
X_normalized_transformed = transformer.fit_transform(X_normalized)

plt.scatter(X_normalized_transformed[:,0], X_normalized_transformed[:,1], c=color_values)
plt.show()

print(X_normalized_transformed)