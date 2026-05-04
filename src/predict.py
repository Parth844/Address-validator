import re
import joblib
import numpy as np
from difflib import get_close_matches
from .pincode_lookup import validate_pincode_against_address, lookup_pincode, lookup_pincode_districts, ALL_DISTRICTS
from src.features import extract_features
from src.state_list import STATE_LIST
model      = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Known Indian pincodes by state prefix — first 2 digits
PINCODE_STATE_PREFIXES = {
    "11": "Delhi",
    "12": "Haryana", "13": "Haryana",
    "14": "Punjab",  "15": "Punjab",
    "16": "Punjab/Chandigarh",
    "17": "Himachal Pradesh",
    "18": "Jammu and Kashmir", "19": "Jammu and Kashmir",
    "20": "Uttar Pradesh", "21": "Uttar Pradesh", "22": "Uttar Pradesh",
    "23": "Uttar Pradesh", "24": "Uttar Pradesh", "25": "Uttar Pradesh",
    "26": "Uttar Pradesh", "27": "Uttar Pradesh", "28": "Uttar Pradesh",
    "30": "Rajasthan", "31": "Rajasthan", "32": "Rajasthan",
    "33": "Rajasthan", "34": "Rajasthan",
    "36": "Gujarat",   "37": "Gujarat",   "38": "Gujarat",   "39": "Gujarat",
    "40": "Maharashtra", "41": "Maharashtra", "42": "Maharashtra",
    "43": "Maharashtra", "44": "Maharashtra",
    "45": "Madhya Pradesh", "46": "Madhya Pradesh", "47": "Madhya Pradesh",
    "48": "Madhya Pradesh", "49": "Chhattisgarh",
    "50": "Telangana/Andhra Pradesh", "51": "Andhra Pradesh",
    "52": "Andhra Pradesh", "53": "Andhra Pradesh",
    "56": "Karnataka",  "57": "Karnataka", "58": "Karnataka", "59": "Karnataka",
    "60": "Tamil Nadu", "61": "Tamil Nadu", "62": "Tamil Nadu",
    "63": "Tamil Nadu", "64": "Tamil Nadu",
    "67": "Kerala",    "68": "Kerala",    "69": "Kerala",
    "70": "West Bengal", "71": "West Bengal", "72": "West Bengal",
    "73": "West Bengal", "74": "West Bengal",
    "75": "Odisha",    "76": "Odisha",    "77": "Odisha",
    "78": "Assam",     "79": "Northeast",
    "80": "Bihar",     "81": "Bihar",     "82": "Bihar",     "83": "Jharkhand",
    "84": "Bihar",     "85": "Bihar",
    "90": "Army Post Office", "91": "Army Post Office",
}

# State → city name fragments (lowercase) for cross-validation
STATE_CITY_HINTS = {
    "maharashtra":     ["mumbai","pune","nagpur","nashik","thane","aurangabad","solapur"],
    "karnataka":       ["bangalore","bengaluru","mysore","hubli","mangalore","belgaum"],
    "tamil nadu":      ["chennai","coimbatore","madurai","trichy","salem","vellore","erode"],
    "uttar pradesh":   ["lucknow","kanpur","agra","varanasi","allahabad","meerut","noida","ghaziabad"],
    "delhi":           ["delhi","dwarka","rohini","janakpuri","saket","karol bagh","pitampura"],
    "gujarat":         ["ahmedabad","surat","vadodara","rajkot","gandhinagar","bhavnagar"],
    "rajasthan":       ["jaipur","jodhpur","udaipur","kota","bikaner","ajmer","alwar"],
    "west bengal":     ["kolkata","howrah","durgapur","asansol","siliguri"],
    "telangana":       ["hyderabad","warangal","nizamabad","karimnagar"],
    "andhra pradesh":  ["visakhapatnam","vijayawada","guntur","nellore","kurnool","tirupati"],
    "kerala":          ["thiruvananthapuram","kochi","kozhikode","thrissur","kollam"],
    "punjab":          ["ludhiana","amritsar","jalandhar","patiala","bathinda"],
    "haryana":         ["faridabad","gurgaon","gurugram","panipat","ambala","rohtak"],
    "madhya pradesh":  ["bhopal","indore","jabalpur","gwalior","ujjain"],
    "bihar":           ["patna","gaya","bhagalpur","muzaffarpur"],
    "odisha":          ["bhubaneswar","cuttack","rourkela","puri"],
    "jharkhand":       ["ranchi","jamshedpur","dhanbad","bokaro"],
    "chhattisgarh":    ["raipur","bhilai","bilaspur","korba"],
    "assam":           ["guwahati","silchar","dibrugarh"],
    "chandigarh":      ["chandigarh"],
    "goa":             ["panaji","margao","vasco"],
    "himachal pradesh":["shimla","manali","dharamsala"],
    "uttarakhand":     ["dehradun","haridwar","roorkee","haldwani"],
    "jammu and kashmir":["srinagar","jammu","anantnag","baramulla"],
}

# Characters/patterns that are definitely garbage
GARBAGE_PATTERNS = [
    r"^[^a-zA-Z0-9]+$",                      # only symbols
    r"^[a-zA-Z]{1,3}\s?[a-zA-Z]{1,3}$",      # e.g. "ab cd"
    r"^\d+$",                                  # only digits
    r"[!@#$%^&*]{2,}",                         # 2+ special chars together
    r"^(na|n/a|null|none|test|address|addr|xyz|abc|sample|dummy)$",  # placeholders
    r"(.)\1{4,}",                              # 5+ repeated chars: "aaaaa"
]

NOISE_ONLY_RE    = re.compile(r"^[\W_\s]+$")
PINCODE_RE       = re.compile(r"\b(\d{6})\b")
PARTIAL_PIN_RE   = re.compile(r"\b\d{4,5}\b")        # 4 or 5 digit "pin"
ALPHAPIN_RE      = re.compile(r"\b[A-Z0-9]{5,7}\b")  # like "4000AB"

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-CORRECTION — fix common formatting mistakes before rule evaluation
# Returns (corrected_address, list_of_corrections_applied)
# ─────────────────────────────────────────────────────────────────────────────

_COMMA_PIN_RE   = re.compile(r"\b(\d{3}),(\d{3})\b")   # 160,015  → 160015
_DASH_PIN_RE    = re.compile(r"\b(\d{3})-(\d{3})\b")   # 160-015  → 160015
_SPACE_PIN_RE   = re.compile(r"\b(\d{3})\s(\d{3})\b")  # 160 015  → 160015
_WORD_PIN_RE    = re.compile(r"([a-zA-Z])(\d{6})\b")   # pradesh201002 → pradesh 201002
_MULTI_SPACE_RE = re.compile(r"  +")                    # 2+ spaces → single
_TRAILING_COMMA = re.compile(r",\s*,")                  # double commas

_STATE_FIXES = {
    "uttarpradesh": "uttar pradesh",
    "madhyapradesh": "madhya pradesh",
    "andhrapradesh": "andhra pradesh",
    "himachalpradesh": "himachal pradesh",
    "arunachalpradesh": "arunachal pradesh",
    "tamilnadu": "tamil nadu",
    "westbengal": "west bengal",
    "newdelhi": "new delhi",
    "jammuandkashmir": "jammu and kashmir"
}

_PROPERTY_IDENTIFIERS_RE = re.compile(
    r"\b(ho|house|hse|h\.?|flat|flt|f\.?|plot|plt|p\.?|phase|ph\.?|room|r\.?|door|d\.?|shop|sh\.?|office|off\.?|unit|u\.?|ward|w\.?)\s*no\b\.?",
    re.IGNORECASE
)

def _auto_correct(address: str) -> tuple[str, list[str]]:
    """
    Apply safe, deterministic corrections to common formatting mistakes.
    Returns the cleaned address and a list of human-readable fix descriptions.
    Only fixes things we are highly confident about — never changes words.
    """
    original   = address
    fixes      = []
    corrected  = address

    # 1. Pincode with comma inside: 160,015 → 160015
    if _COMMA_PIN_RE.search(corrected):
        corrected = _COMMA_PIN_RE.sub(r"\1\2", corrected)
        fixes.append("removed comma inside pincode (e.g. 160,015 → 160015)")

    # 2. Pincode with dash inside: 160-015 → 160015
    #    But don't touch house numbers like 23-4 (they are short: d{1,3}-d{1,2})
    if _DASH_PIN_RE.search(corrected):
        corrected = _DASH_PIN_RE.sub(r"\1\2", corrected)
        fixes.append("removed dash inside pincode (e.g. 160-015 → 160015)")

    # 3. Pincode split by space: "160 015" → "160015"
    #    Only when the result is a valid 6-digit sequence (no leading zero)
    def _fix_space_pin(m):
        merged = m.group(1) + m.group(2)
        if not merged.startswith("0"):
            fixes.append(f"merged space-separated pincode ({m.group()} → {merged})")
            return merged
        return m.group()
    corrected = _SPACE_PIN_RE.sub(_fix_space_pin, corrected)

    # 4. Pincode merged with text
    if _WORD_PIN_RE.search(corrected):
        corrected = _WORD_PIN_RE.sub(r"\1 \2", corrected)
        fixes.append("separated pincode from word")

    # 4.5 State missing spaces
    for bad, good in _STATE_FIXES.items():
        pattern = re.compile(rf"\b{bad}\b", re.IGNORECASE)
        if pattern.search(corrected):
            corrected = pattern.sub(good, corrected)
            if f"added space to '{good}'" not in fixes:
                fixes.append(f"added space to '{good}'")

    # 4.6 Normalize house/property identifiers
    if _PROPERTY_IDENTIFIERS_RE.search(corrected):
        corrected = _PROPERTY_IDENTIFIERS_RE.sub("h.no.", corrected)
        if "normalized house/property identifier" not in fixes:
            fixes.append("normalized house/property identifier")

    # 4. Collapse multiple spaces
    single_spaced = _MULTI_SPACE_RE.sub(" ", corrected).strip()
    if single_spaced != corrected:
        fixes.append("collapsed extra whitespace")
        corrected = single_spaced

    # 5. Remove double commas
    clean_commas = _TRAILING_COMMA.sub(",", corrected)
    if clean_commas != corrected:
        fixes.append("removed duplicate commas")
        corrected = clean_commas

    # 6. Strip leading/trailing punctuation artifacts
    stripped = corrected.strip(" ,;")
    if stripped != corrected:
        fixes.append("removed leading/trailing punctuation")
        corrected = stripped

    return corrected, fixes
HOUSE_RE         = re.compile(
    r"\b(flat|f-|apt|apartment|house|h\.?\s*no|h no|door no|d\.?no|"
    r"plot|plot no|khasra|survey|block|tower|wing|unit|shop|office|"
    r"floor|villa|bungalow|room|r\.no|no\.)\b",
    re.IGNORECASE
)
STREET_RE        = re.compile(
    r"\b(road|rd|street|st|marg|lane|ln|avenue|ave|nagar|ngr|colony|"
    r"enclave|vihar|chowk|sector|sec|bazaar|market|layout|extension|"
    r"phase|block|cross|main|circle|square|path)\b",
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Collapse extra whitespace, strip, lowercase."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def _confidence_band(prob: float) -> str:
    if prob >= 0.85:  return "HIGH"
    if prob >= 0.65:  return "MEDIUM"
    if prob >= 0.45:  return "LOW"
    return "VERY LOW"

# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE — each rule returns (triggered: bool, reason: str, penalty: float)
# penalty: how much to subtract from the final ML score (0.0 – 1.0)
# ─────────────────────────────────────────────────────────────────────────────

def rule_garbage(text: str, raw: str):
    """Completely nonsensical input."""
    stripped = raw.strip()
    if NOISE_ONLY_RE.match(stripped):
        return True, "Input contains only noise/symbols", 1.0
    for pat in GARBAGE_PATTERNS:
        if re.search(pat, stripped, re.IGNORECASE):
            return True, f"Garbage pattern detected: '{stripped[:40]}'", 1.0
    # very short address (< 10 chars) with no digits
    if len(stripped) < 10 and not re.search(r"\d", stripped):
        return True, "Address too short to be valid", 0.9
    return False, "", 0.0

def rule_pincode_prefix_mismatch(text: str, raw: str):
    """
    Validate pincode using first 2 digits (India Post zone logic).
    Acts as fallback if DB/API misses.
    """
    pins = PINCODE_RE.findall(text)
    if not pins:
        return False, "", 0.0

    pin = pins[0]
    prefix = pin[:2]

    expected_state = PINCODE_STATE_PREFIXES.get(prefix)
    if not expected_state:
        return False, "", 0.0

    detected_states = _detect_states_fuzzy(text)

    if detected_states:
        # normalize for comparison
        expected_state_l = expected_state.lower()

        # handle combined states like "telangana/andhra pradesh"
        expected_options = [s.strip() for s in expected_state_l.split("/")]

        if not any(ds.lower() in expected_options for ds in detected_states):
            return True, (
                f"Pincode {pin} belongs to {expected_state} (by prefix {prefix}), "
                f"but address mentions {detected_states}"
            ), 0.85

    return False, "", 0.0


def rule_missing_pincode(text: str, raw: str):
    """No 6-digit pincode present."""
    if not PINCODE_RE.search(text):
        # check if a partial pin exists (user typed 5 digits)
        if PARTIAL_PIN_RE.search(text):
            return True, "Pincode appears incomplete (4–5 digits)", 0.75
        if ALPHAPIN_RE.search(raw.upper()):
            return True, "Pincode appears alphanumeric (invalid format)", 0.85
        return True, "Missing 6-digit pincode", 0.80
    return False, "", 0.0


def rule_invalid_pincode_range(text: str, raw: str):
    """Pincode outside valid Indian range (100000–999999) or starting with 0."""
    pins = PINCODE_RE.findall(text)
    for pin in pins:
        if pin.startswith("0"):
            return True, f"Pincode {pin} starts with 0 (invalid)", 0.90
        if int(pin) < 100000:
            return True, f"Pincode {pin} out of valid Indian range", 0.85
    return False, "", 0.0


def rule_multiple_states(text: str, raw: str):
    """More than one Indian state found → conflicting address."""
    found = _detect_states_fuzzy(text)
    if len(found) > 1:
        return True, f"Conflicting states detected: {found}", 0.95
    return False, "", 0.0


def rule_state_pincode_mismatch(text: str, raw: str):
    pins = PINCODE_RE.findall(text)
    if not pins:
        return False, "", 0.0

    pin = pins[0]
    result = validate_pincode_against_address(pin, text) or {}

    if not result.get("pincode_exists", True):
        return True, f"Pincode {pin} does not exist in India Post database.", 0.95

    db_states = list(set(result.get("db_states") or []))

    if not result.get("state_match", True):
        return True, (
            f"Pincode {pin} belongs to {db_states} per India Post, "
            f"but a different state was found in the address."
        ), 0.80, "PINCODE_MISMATCH"

    return False, "", 0.0

COMMON_SYNONYMS = {
    "bangalore": "bengaluru",
    "bengaluru": "bangalore",
    "gurgaon": "gurugram",
    "gurugram": "gurgaon",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "orissa": "odisha",
    "trivandrum": "thiruvananthapuram",
    "cochin": "kochi",
    "baroda": "vadodara",
    "waltair": "visakhapatnam",
    "vizag": "visakhapatnam",
    "pondicherry": "puducherry"
}

def rule_city_pincode_mismatch(text: str, raw: str):
    """
    Check if the city or district mentioned in the address matches the pincode's district.
    """
    pins = PINCODE_RE.findall(text)
    if not pins:
        return False, "", 0.0

    pin = pins[0]
    db_districts = lookup_pincode_districts(pin)
    if not db_districts:
        return False, "", 0.0

    # find which of our ALL_DISTRICTS are mentioned in the address
    detected_locations = set()
    for loc in ALL_DISTRICTS:
        if loc in text:
            detected_locations.add(loc)
            # Also add its known synonyms!
            if loc in COMMON_SYNONYMS:
                detected_locations.add(COMMON_SYNONYMS[loc])

    # If the address doesn't mention any city or district from the database,
    # then there is no conflict! (User might just not have written the city).
    if not detected_locations:
        return False, "", 0.0

    # Let's normalize the districts from the database as well to check matches.
    valid_locations = set(db_districts)
    for dist in db_districts:
        if dist in COMMON_SYNONYMS:
            valid_locations.add(COMMON_SYNONYMS[dist])

    # If any detected location matches any valid location for the pincode, then no mismatch.
    if any(loc in valid_locations for loc in detected_locations):
        return False, "", 0.0

    # Otherwise, the user mentioned a different city/district than the pincode belongs to.
    detected_list = sorted(list(detected_locations))[:3]
    valid_list = sorted(list(db_districts))[:3]
    return True, (
        f"City/district '{detected_list[0]}' found in address, but pincode {pin} "
        f"belongs to {valid_list}."
    ), 0.80, "PINCODE_MISMATCH"

def rule_missing_house_number(text: str, raw: str):
    """No house/flat/plot identifier at all."""
    if not HOUSE_RE.search(text) and not re.search(r"\b\d+[a-z]?\b", text):
        return True, "No house/flat/plot number found", 0.50
    return False, "", 0.0


def rule_missing_street_or_locality(text: str, raw: str):
    """No street/locality/area keyword."""
    if not STREET_RE.search(text):
        # give it a pass if address is reasonably long (may use proper names)
        word_count = len(text.split())
        if word_count < 5:
            return True, "No street/locality identifier and address is very short", 0.45
    return False, "", 0.0


def rule_too_many_digits(text: str, raw: str):
    """Input is mostly numeric (likely just numbers)."""
    digits = sum(c.isdigit() for c in raw)
    total  = max(len(raw.replace(" ", "")), 1)
    if total < 4:
        return False, "", 0.0
    ratio = digits / total
    if ratio > 0.75 and len(raw.strip()) < 15:
        return True, "Input appears to be mostly numbers (not an address)", 0.90
    return False, "", 0.0


def rule_repeated_state_city(text: str, raw: str):
    """Same city/state word repeated (data entry error)."""
    words = [w for w in text.split() if len(w) > 4]
    if len(words) != len(set(words)):
        from collections import Counter
        dupes = [w for w, c in Counter(words).items() if c > 1]
        if dupes:
            return True, f"Repeated location tokens detected: {dupes}", 0.55
    return False, "", 0.0


def rule_no_alpha(text: str, raw: str):
    """No alphabetic characters at all."""
    if not re.search(r"[a-zA-Z]", raw):
        return True, "No alphabetic characters in input", 1.0
    return False, "", 0.0


def rule_suspicious_length(text: str, raw: str):
    """Extremely short or absurdly long address."""
    l = len(raw.strip())
    if l < 15:
        return True, f"Address suspiciously short ({l} chars)", 0.70
    if l > 400:
        return True, f"Address suspiciously long ({l} chars) — possible junk", 0.40
    return False, "", 0.0


def rule_pincode_comma_glitch(text: str, raw: str):
    """Pincode split by comma: '560,001'."""
    if re.search(r"\b\d{3},\d{3}\b", raw):
        return True, "Pincode appears to have a misplaced comma (e.g. '560,001')", 0.65
    return False, "", 0.0


def rule_state_city_cross_check(text: str, raw: str):
    """City mentioned belongs to a different state than stated."""
    states = _detect_states_fuzzy(text)
    if not states:
        return False, "", 0.0
    state = states[0]
    hints = STATE_CITY_HINTS.get(state, [])
    if not hints:
        return False, "", 0.0

    # cities of OTHER states
    other_cities = []
    for s, cities in STATE_CITY_HINTS.items():
        if s != state:
            other_cities.extend(cities)

    tokens = _token_set(text)
    conflicting = [c for c in other_cities if c in tokens and c not in hints]
    if conflicting:
        return True, (
            f"City '{conflicting[0]}' does not belong to state '{state}'"
        ), 0.75
    return False, "", 0.0


# All rules in priority order (hard blocks first, soft penalties later)
# Note: pincode formatting glitches (comma/dash/space in pin) are handled
# upstream by _auto_correct() before rules run — no rule needed here.
RULES = [
    rule_garbage,
    rule_no_alpha,
    rule_too_many_digits,
    rule_suspicious_length,
    rule_missing_pincode,
    rule_invalid_pincode_range,

    rule_multiple_states,
    rule_state_pincode_mismatch,     # ✅ FIRST (accurate)
    rule_city_pincode_mismatch,

    #rule_pincode_prefix_mismatch,    # ✅ fallback

    rule_state_city_cross_check,
    rule_missing_house_number,
    rule_missing_street_or_locality,
    rule_repeated_state_city,
]
# ─────────────────────────────────────────────────────────────────────────────
# STATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_states_fuzzy(text: str) -> list:
    """Return list of matched Indian states (exact + fuzzy) from lowercased text."""
    words   = text.split()
    detected = set()

    for state in STATE_LIST:
        state_l = state.lower()
        if state_l in text:
            detected.add(state)
            continue
        # multi-word state: check if all tokens present
        state_tokens = state_l.split()
        if len(state_tokens) > 1:
            if all(t in text for t in state_tokens):
                detected.add(state)
                continue
        # single-word fuzzy
        for word in words:
            if len(word) < 4:
                continue
            if get_close_matches(word, [state_l], n=1, cutoff=0.82):
                detected.add(state)
                break

    return list(detected)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict(address: str) -> dict:
    raw  = str(address)

    # ── Step 1: Auto-correct common formatting mistakes ───────────────────────
    corrected, corrections = _auto_correct(raw)
    was_corrected = corrected != raw

    # Work on corrected version from here on
    text = _normalise(corrected)


    # ── Step 2: Run rule engine on corrected address ──────────────────────────
    triggered_rules = []
    total_penalty   = 0.0
    is_critical     = False
    critical_reason = ""
    rule_warnings = []

    total_penalty = 0.0

    for rule_fn in RULES:
        result = rule_fn(text, corrected)

        if len(result) == 4:
            fired, reason, penalty, warning = result
        else:
            fired, reason, penalty = result
            warning = None

        if fired:
            triggered_rules.append({
                "rule": rule_fn.__name__,
                "reason": reason,
                "penalty": penalty
            })

            if warning:
                rule_warnings.append(warning)
        
            # multiplicative stacking
            total_penalty = 1 - ((1 - total_penalty) * (1 - penalty))

    

    # ── Step 3: Hard block BEFORE ML ──
    if is_critical:
        return {
            "address": raw,
            "corrected_address": corrected if was_corrected else None,
            "corrections": corrections,
            "valid_score": 0.0,
            "ml_raw_score": None,
            "prediction": "INVALID",
            "confidence": "HIGH",
            "reason": critical_reason,
            "rules_triggered": triggered_rules,
            "source": "rule_engine_only",
        }

# ── Step 4: ML inference on corrected address ────────────────────────────
    X_tfidf = vectorizer.transform([corrected])

    # Define X_num FIRST
    X_num = np.array(extract_features(corrected)).reshape(1, -1)

    # Then combine
    X = np.hstack([X_tfidf.toarray(), X_num])

    # Predict
    proba   = model.predict_proba(X)[0]
    ml_prob = float(proba[1])

    adjusted_prob = ml_prob * (1 - 0.6 * total_penalty)
    adjusted_prob = max(0.0, min(1.0, adjusted_prob))



    # Slightly higher threshold when any soft rule fired
    if total_penalty > 0.6:
        threshold = 0.72
    elif total_penalty > 0.3:
        threshold = 0.68
    else:
        threshold = 0.60


    prediction = "VALID" if adjusted_prob >= threshold else "INVALID"

    # Build reason string
    if not triggered_rules and not corrections:
        reason = "ML prediction — no issues found"
    elif corrections and not triggered_rules:
        reason = "Auto-corrected formatting issues; ML passed. Fixes: " + "; ".join(corrections)
    else:
        rule_reasons = "; ".join(r["reason"] for r in triggered_rules)
        reason = f"ML adjusted by rule violations: {rule_reasons}"
        if corrections:
            reason += f". Auto-corrections applied: {'; '.join(corrections)}"

    return {
        "address": raw,
        "corrected_address": corrected if was_corrected else None,
        "corrections": corrections,
        "valid_score": round(adjusted_prob, 4),
        "ml_raw_score": round(ml_prob, 4),
        "rule_penalty": round(total_penalty, 4),
        "prediction": prediction,
        "confidence": _confidence_band(adjusted_prob),
        "reason": reason,
        "rules_triggered": triggered_rules,
        "warnings": rule_warnings,   # ✅ ADD THIS
        "source": "ml+rules",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(addresses: list) -> list:
    """Predict a list of addresses and return list of result dicts."""
    return [predict(addr) for addr in addresses]


# ─────────────────────────────────────────────────────────────────────────────
# CLI / QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    samples = [
        # ── VALID ──
        "Flat 204, Sunrise Apartments, MG Road, Bangalore, Karnataka 560001",
        "Plot No 78, Sector 15, Chandigarh 160015",
        "45 Gandhi Nagar, Near City Hospital, Mumbai, Maharashtra 400001",
        "H No 23/4, Civil Lines, Allahabad, Uttar Pradesh 211001",
        "gram mahanagar, dist. siliguri, west bengal - 734001",
        # ── INVALID ──
        "asdfgh jklmnop",
        "Mumbai 4000AB",
        "Delhi 110001",                                    # missing house+street
        "Ghaziabad Haryana Uttar Pradesh 201001",          # dual state
        "560001",
        "!!! ### ???",
        "MG Road Bangalore",                               # partial
        "Bangalor Karnatak 56001",                         # typos + short pin
        "H No 12, Koramangala, Bangalore, Tamil Nadu 560034",  # wrong state-city
        "Flat 5, Sector 22, Chandigarh 160,015",           # comma in pin
    ]

    for addr in samples:
        result = predict(addr)
        print(json.dumps(result, indent=2))
        print("-" * 60)