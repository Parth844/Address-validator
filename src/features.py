import re
import numpy as np
import pandas as pd

# =========================
# LOAD GEO DATA (once)
# =========================
df = pd.read_csv("data/pincode_clean.csv")

# Create lookup
PINCODE_TO_STATE = dict(zip(df["pincode"].astype(str), df["state"]))

# Unique states list
STATE_LIST = list(
    set(
        df["state"]
        .dropna()                 # 🚨 remove NaN
        .astype(str)
        .str.lower()
        .str.strip()
    )
)
print("done loading geo data")
# =========================
# FEATURE FUNCTION
# =========================
def extract_features(text):
    text = str(text).lower()

    # -------------------------
    # 1. PINCODE FEATURES
    # -------------------------
    pincode_match = re.search(r"\b\d{6}\b", text)
    has_pincode = int(bool(pincode_match))

    pincode_valid = 0
    state_match = 0

    if pincode_match:
        pincode = pincode_match.group()

        if pincode in PINCODE_TO_STATE:
            pincode_valid = 1
            mapped_state = PINCODE_TO_STATE[pincode]

            if mapped_state in text:
                state_match = 1

    # -------------------------
    # 2. STATE FEATURES
    # -------------------------
    found_states = [s for s in STATE_LIST if isinstance(s, str) and s in text]
    has_state = int(len(found_states) > 0)
    multiple_states = int(len(found_states) > 1)

    # -------------------------
    # 3. BASIC TEXT FEATURES
    # -------------------------
    has_numbers = int(bool(re.search(r"\d+", text)))
    special_chars = len(re.findall(r"[^\w\s]", text))
    text_length = len(text.split())

    # -------------------------
    # 4. STREET KEYWORDS
    # -------------------------
    keywords = ["road", "rd", "street", "st", "nagar", "sector", "colony","lane", "block", "phase", "area", "village"]
    has_street = int(any(k in text for k in keywords))

    # -------------------------
    # FINAL FEATURE VECTOR
    # -------------------------
    return np.array([
        has_pincode,
        pincode_valid,
        state_match,
        has_state,
        multiple_states,
        has_numbers,
        has_street,
        special_chars,
        text_length
    ])
