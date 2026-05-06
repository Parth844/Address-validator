import pandas as pd
import os

# Load once
DATA_PATH = os.path.join("data", "pincode_clean.csv")
df = pd.read_csv(DATA_PATH, dtype={"pincode": str})

# Create lookup
PINCODE_DB = df.groupby("pincode")["state"].apply(list).to_dict()

PINCODE_TO_CITIES = {}
ALL_LOCATIONS = set()

df["city"] = df["city"].fillna("").astype(str).str.lower().str.strip()
df["district"] = df["district"].fillna("").astype(str).str.lower().str.strip()
df["pincode"] = df["pincode"].fillna("").astype(str).str.strip()

def clean_location(name):
    """Remove ' b.o', ' s.o', ' h.o' and extra dots/spaces."""
    name = name.replace(" b.o", "").replace(" s.o", "").replace(" h.o", "")
    name = name.replace(".bo", "").replace(".so", "").replace(".ho", "")
    return name.strip()

for p, c, d in zip(df["pincode"], df["city"], df["district"]):
    if not p:
        continue
    if p not in PINCODE_TO_CITIES:
        PINCODE_TO_CITIES[p] = set()
    
    # Clean city name (which contains office name in this dataset)
    cleaned_c = clean_location(c)
    if cleaned_c:
        PINCODE_TO_CITIES[p].add(cleaned_c)
        ALL_LOCATIONS.add(cleaned_c)
    
    # Also keep district as a fallback location
    if d:
        PINCODE_TO_CITIES[p].add(d)
        ALL_LOCATIONS.add(d)

ADDITIONAL_CITIES = {
    "mumbai", "pune", "nagpur", "nashik", "thane", "aurangabad", "solapur",
    "bangalore", "bengaluru", "mysore", "hubli", "mangalore", "belgaum",
    "chennai", "coimbatore", "madurai", "trichy", "salem", "vellore", "erode",
    "lucknow", "kanpur", "agra", "varanasi", "allahabad", "meerut", "noida", "ghaziabad",
    "delhi", "dwarka", "rohini", "janakpuri", "saket", "karol bagh", "pitampura",
    "ahmedabad", "surat", "vadodara", "rajkot", "gandhinagar", "bhavnagar",
    "jaipur", "jodhpur", "udaipur", "kota", "bikaner", "ajmer", "alwar",
    "kolkata", "howrah", "durgapur", "asansol", "siliguri",
    "hyderabad", "warangal", "nizamabad", "karimnagar",
    "visakhapatnam", "vijayawada", "guntur", "nellore", "kurnool", "tirupati",
    "thiruvananthapuram", "kochi", "kozhikode", "thrissur", "kollam",
    "ludhiana", "amritsar", "jalandhar", "patiala", "bathinda",
    "faridabad", "gurgaon", "gurugram", "panipat", "ambala", "rohtak",
    "bhopal", "indore", "jabalpur", "gwalior", "ujjain",
    "patna", "gaya", "bhagalpur", "muzaffarpur",
    "bhubaneswar", "cuttack", "rourkela", "puri",
    "ranchi", "jamshedpur", "dhanbad", "bokaro",
    "raipur", "bhilai", "bilaspur", "korba",
    "guwahati", "silchar", "dibrugarh",
    "chandigarh", "panaji", "margao", "vasco",
    "shimla", "manali", "dharamsala",
    "dehradun", "haridwar", "roorkee", "haldwani",
    "srinagar", "jammu", "anantnag", "baramulla"
}

for c in ADDITIONAL_CITIES:
    ALL_LOCATIONS.add(c)

CITY_STATE_TO_PINCODE = {}
df["state"] = df["state"].fillna("").astype(str).str.lower().str.strip()
for p, d, s in zip(df["pincode"], df["district"], df["state"]):
    if not p or not d or not s:
        continue
    for term in [d] + [w for w in d.split() if w not in {"urban", "rural", "east", "west", "north", "south", "central"}]:
        if len(term) >= 3:
            key = (term, s)
            if key not in CITY_STATE_TO_PINCODE:
                CITY_STATE_TO_PINCODE[key] = []
            if p not in CITY_STATE_TO_PINCODE[key]:
                CITY_STATE_TO_PINCODE[key].append(p)

def lookup_pincode(pincode: str):
    """
    Returns list of states for a pincode
    """
    return PINCODE_DB.get(pincode, [])


def lookup_pincode_districts(pincode: str):
    """
    Returns set of cities/districts for a pincode
    """
    return PINCODE_TO_CITIES.get(pincode, set())


def validate_pincode_against_address(pincode: str, address: str):
    """
    Validate if pincode exists and matches state in address
    """
    address = address.lower()

    states = lookup_pincode(pincode)

    if not states:
        return {
            "pincode_exists": False,
            "state_match": None,
            "db_states": []
        }

    # check if any state matches address text
    match = any(state.lower() in address for state in states)

    return {
        "pincode_exists": True,
        "state_match": match,
        "db_states": states
    }