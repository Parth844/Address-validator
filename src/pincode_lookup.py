import pandas as pd
import os

# Load once
DATA_PATH = os.path.join("data", "pincode_clean.csv")
df = pd.read_csv(DATA_PATH, dtype={"pincode": str})

# Create lookup
PINCODE_DB = df.groupby("pincode")["state"].apply(list).to_dict()


def lookup_pincode(pincode: str):
    """
    Returns list of states for a pincode
    """
    return PINCODE_DB.get(pincode, [])


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