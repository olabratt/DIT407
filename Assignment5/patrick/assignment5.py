import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
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

X_train, y_train, X_test, y_test = train_test_split(X_normalized, y, test_size = 0.2, random_state=1234)

# Problem 2

inertia_values = []
k_values = [i for i in range(1, 10)]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=SEED).fit(X_normalized)
    inertia_values.append(kmeans.inertia_)

plt.grid()
plt.plot(k_values, inertia_values, 'o-')
plt.show()

print(X_normalized)