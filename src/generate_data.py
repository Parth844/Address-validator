import random
import pandas as pd

cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]
states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana"]
streets = ["MG Road", "Nehru Street", "Gandhi Nagar", "Park Street", "Sector 15"]

def generate_valid():
    return f"{random.randint(1,999)}, {random.choice(streets)}, {random.choice(cities)}, {random.choice(states)} {random.randint(100000,999999)}"

def generate_invalid():
    samples = [
        "asdfgh jkl",
        "123 random text",
        "Mumbai 4000AB",
        "No city no state",
        "!!! ??? ###",
        f"{random.choice(cities)} {random.randint(1000,9999)}"
    ]
    return random.choice(samples)

data = []

for _ in range(5000):
    data.append((generate_valid(), 1))
    data.append((generate_invalid(), 0))

df = pd.DataFrame(data, columns=["address", "label"])
df.to_csv("data/raw_addresses.csv", index=False)

print("Dataset created!")