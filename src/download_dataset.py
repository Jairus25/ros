from ucimlrepo import fetch_ucirepo

# Download the real Gas Sensor Array Drift Dataset.
dataset = fetch_ucirepo(id=224)

X = dataset.data.features
y = dataset.data.targets

print("\nDataset downloaded successfully.")

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget columns:")
print(y.columns.tolist())

print("\nFirst five feature rows:")
print(X.head())

print("\nFirst five target rows:")
print(y.head())

print("\nClass distribution:")
for column in y.columns:
    print(y[column].value_counts(dropna=False))
