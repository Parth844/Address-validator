import pandas as pd

df = pd.read_csv("data/pincode_raw.csv")

# Standardize column names
df.columns = df.columns.str.lower()

# Rename for clarity
df = df.rename(columns={
    "officename": "city",
    "statename": "state"
})

# Clean text
for col in ["city", "district", "state"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Convert pincode to string
df["pincode"] = df["pincode"].astype(str)

# Remove duplicates
df = df.drop_duplicates(subset=["pincode", "city", "district", "state"])

# Keep only useful columns
df = df[["pincode", "city", "district", "state"]]

# Save clean dataset
df.to_csv("data/pincode_clean.csv", index=False)

print("✅ Clean dataset ready:", df.shape)