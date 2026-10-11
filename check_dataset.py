import pandas as pd

df = pd.read_csv("incident_event_log.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nSLA Values:")
print(df["made_sla"].value_counts())