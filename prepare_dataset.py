import pandas as pd

df = pd.read_csv("incident_event_log.csv")

print("Original rows:", len(df))
print("Unique incidents:", df["number"].nunique())

df = df.sort_values(["number", "opened_at"])

features = [
    "number",
    "impact",
    "urgency",
    "priority",
    "reassignment_count",
    "reopen_count",
    "sys_mod_count",
    "knowledge",
    "u_priority_confirmation",
    "category",
    "subcategory",
    "contact_type",
    "assignment_group",
    "made_sla"
]

df = df[features]

df = df.groupby("number").agg({
    "impact": "first",
    "urgency": "first",
    "priority": "first",
    "reassignment_count": "max",
    "reopen_count": "max",
    "sys_mod_count": "max",
    "knowledge": "first",
    "u_priority_confirmation": "first",
    "category": "first",
    "subcategory": "first",
    "contact_type": "first",
    "assignment_group": "first",
    "made_sla": "last"
}).reset_index()

df["sla_breach"] = (~df["made_sla"]).astype(int)

df = df.drop(columns=["made_sla"])

print("\nPrepared dataset shape:", df.shape)

print("\nSLA Breach distribution:")
print(df["sla_breach"].value_counts())

df.to_csv("prepared_incident_data.csv", index=False)

print("\nSaved as: prepared_incident_data.csv")