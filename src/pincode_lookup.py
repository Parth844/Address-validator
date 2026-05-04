import pandas as pd
import os

# Load once
DATA_PATH = os.path.join("data", "pincode_clean.csv")
df = pd.read_csv(DATA_PATH, dtype={"pincode": str})

# Create lookup
PINCODE_DB = df.groupby("pincode")["state"].apply(list).to_dict()

PINCODE_TO_DISTRICTS = {}
ALL_DISTRICTS = set()

df["district"] = df["district"].fillna("").astype(str).str.lower().str.strip()
df["pincode"] = df["pincode"].fillna("").astype(str).str.strip()

for p, d in zip(df["pincode"], df["district"]):
    if not p or not d:
        continue
    if p not in PINCODE_TO_DISTRICTS:
        PINCODE_TO_DISTRICTS[p] = set()
    PINCODE_TO_DISTRICTS[p].add(d)
    ALL_DISTRICTS.add(d)

    # Add individual words if district has multiple words, except for stopwords
    words = [w for w in d.split() if w not in {"urban", "rural", "east", "west", "north", "south", "central"}]
    for w in words:
        if len(w) >= 3:
            ALL_DISTRICTS.add(w)
            PINCODE_TO_DISTRICTS[p].add(w)

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
    ALL_DISTRICTS.add(c)

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
    Returns set of districts for a pincode
    """
    return PINCODE_TO_DISTRICTS.get(pincode, set())


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