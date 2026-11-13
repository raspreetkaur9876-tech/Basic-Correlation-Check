
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display dataset
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# Calculate correlation
correlation = df.corr()

print("\nCorrelation Matrix:")
print(correlation)

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Iris Dataset Correlation Heatmap")
plt.tight_layout()
plt.show()

# Create scatterplots
plt.figure(figsize=(7, 5))
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Sepal Length vs Petal Length")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(df["sepal width (cm)"], df["petal width (cm)"])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.title("Sepal Width vs Petal Width")
plt.show()

# Observations
print("\nObservations:")
print("1. Petal length and petal width have a strong positive correlation.")
print("2. Sepal length and petal length have a positive correlation.")
print("3. Sepal width has a negative correlation with petal length.")
