import pandas as pd

path = "/home/arya/Documents/Pasca Stikom/Resource/Dataset/NCC-2 Dataset Simultaneous Botnet Dataset/all-sensors/sensors-all.binetflow"

df = pd.read_csv(path)

print(df.head())
print("Columns:", df.columns)
print("Shape:", df.shape)
df.to_csv("NCC2AllSensors.csv", index=False)
