"""
Address Validator — Production-ready Streamlit app
"""

import streamlit as st
import json
import time
from src.predict import predict, predict_batch
from src.pincode_lookup import lookup_pincode, lookup_pincode_districts

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Address Validator",
    page_icon="📍",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Indian Address Validator · XGBoost + Rule Engine",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace !important;
}

.block-container { padding-top: 2rem !important; max-width: 720px !important; }

/* Masthead */
.masthead-eyebrow {
    font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
    color: #8a8a85; margin-bottom: .4rem;
}
.masthead-title {
    font-family: 'Fraunces', serif; font-size: 40px; font-weight: 300;
    line-height: 1.1; margin-bottom: .3rem;
}
.masthead-title em { font-style: italic; color: #3b6d11; }
.masthead-sub { font-size: 13px; color: #8a8a85; margin-bottom: 1.5rem; }
.masthead-divider { border: none; border-top: 1px solid #e5e3dc; margin: 0 0 1.5rem; }

/* Verdict banners */
.verdict-banner {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem 1.25rem; border-radius: 6px;
    margin-bottom: .75rem; border: 1px solid;
}
.verdict-banner.valid   { background: #eaf3de; border-color: #97c459; }
.verdict-banner.invalid { background: #fcebeb; border-color: #f09595; }
.verdict-label {
    font-family: 'Fraunces', serif; font-size: 22px;
    font-weight: 600; flex: 1;
}
.verdict-banner.valid   .verdict-label { color: #3b6d11; }
.verdict-banner.invalid .verdict-label { color: #a32d2d; }
.verdict-score {
    font-size: 13px; font-weight: 500; padding: .25rem .65rem; border-radius: 3px;
}
.verdict-banner.valid   .verdict-score { background: #3b6d11; color: #eaf3de; }
.verdict-banner.invalid .verdict-score { background: #a32d2d; color: #fcebeb; }

/* Stat grid */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin: .75rem 0; }
.stat-card {
    background: #f5f4f0; border-radius: 6px; border: 1px solid #e5e3dc;
    padding: .65rem 1rem;
}
.stat-lbl { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: #8a8a85; margin-bottom: .2rem; }
.stat-val { font-size: 18px; font-weight: 500; color: #1a1a18; }

/* Rule items */
.rule-item {
    background: #fff; border: 1px solid #e5e3dc; border-radius: 6px;
    padding: .65rem 1rem; margin-bottom: .4rem;
    display: flex; align-items: flex-start; gap: .75rem;
}
.rule-name { font-weight: 500; color: #1a1a18; white-space: nowrap; font-size: 12px; }
.rule-reason { color: #4a4a47; line-height: 1.5; font-size: 12px; flex: 1; }
.rule-penalty {
    flex-shrink: 0; padding: .15rem .45rem; border-radius: 3px;
    font-size: 10px; font-weight: 500;
    background: #fcebeb; color: #a32d2d; border: 1px solid #f09595;
}

/* Corrections */
.corr-tag {
    display: inline-flex; align-items: center; gap: .3rem;
    padding: .25rem .6rem; border-radius: 3px; font-size: 11px;
    background: #faeeda; color: #854f0b; border: 1px solid #ef9f27; margin: .2rem .2rem 0 0;
}

/* Suggestion card */
.suggest-card {
    background: #e6f1fb; border: 1px solid #185fa5; border-radius: 6px;
    padding: .75rem 1rem; margin-bottom: .5rem;
}
.suggest-type {
    font-size: 10px; letter-spacing: .1em; text-transform: uppercase;
    color: #185fa5; margin-bottom: .3rem; font-weight: 500;
}
.suggest-addr { color: #1a1a18; font-size: 13px; line-height: 1.5; }

/* Section headers */
.sec-head {
    font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
    color: #8a8a85; margin: 1rem 0 .4rem;
}

/* Reason box */
.reason-box {
    background: #fff; border: 1px solid #e5e3dc; border-left: 3px solid #ccc9c0;
    border-radius: 0 6px 6px 0; padding: .75rem 1rem;
    font-size: 13px; color: #4a4a47; line-height: 1.6; margin-bottom: .75rem;
}

/* Badges */
.badge {
    display: inline-flex; align-items: center;
    font-size: 10px; padding: .15rem .45rem; border-radius: 3px;
    border: 1px solid #e5e3dc; color: #8a8a85; margin-right: .3rem;
}

/* Batch result table tweaks */
.batch-valid   { color: #3b6d11; font-weight: 500; }
.batch-invalid { color: #a32d2d; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VALID_EXAMPLES = [
    "Flat 204, Sunrise Apartments, MG Road, Bangalore, Karnataka 560001",
    "Plot No 78, Sector 15, Chandigarh 160015",
    "45 Gandhi Nagar, Near City Hospital, Mumbai, Maharashtra 400001",
    "H No 23/4, Civil Lines, Allahabad, Uttar Pradesh 211001",
    "gram mahanagar, dist. siliguri, west bengal - 734001",
]
INVALID_EXAMPLES = [
    "asdfgh jklmnop",
    "560001",
    "Delhi 110001",
    "!!! ### ???",
    "Bangalor Karnatak 56001",
]
MISMATCH_EXAMPLES = [
    "H No 12, Koramangala, Bangalore, Tamil Nadu 560034",
    "Ghaziabad Haryana Uttar Pradesh 201001",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_verdict(result: dict):
    """Render the main result block."""
    score = result.get("valid_score", 0.0)
    pred  = result.get("prediction", "INVALID")
    is_valid = pred == "VALID"
    cls  = "valid" if is_valid else "invalid"
    icon = "✓" if is_valid else "✕"
    label = "Valid address" if is_valid else "Invalid address"

    st.markdown(f"""
    <div class="verdict-banner {cls}">
        <span style="font-size:20px">{icon}</span>
        <span class="verdict-label">{label}</span>
        <span class="verdict-score">{score*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

    # Progress meter
    st.progress(float(score))

    # Stat grid
    ml_raw   = result.get("ml_raw_score")
    ml_str   = f"{ml_raw*100:.1f}%" if ml_raw is not None else "N/A"
    penalty  = result.get("rule_penalty", 0.0)
    conf     = result.get("confidence", "—")
    source   = result.get("source", "—")

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-lbl">ML raw score</div><div class="stat-val">{ml_str}</div></div>
      <div class="stat-card"><div class="stat-lbl">Rule penalty</div><div class="stat-val">{penalty*100:.1f}%</div></div>
      <div class="stat-card"><div class="stat-lbl">Confidence</div><div class="stat-val">{conf}</div></div>
      <div class="stat-card"><div class="stat-lbl">Source</div><div class="stat-val">{source}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-corrections
    corrections = result.get("corrections") or []
    corrected   = result.get("corrected_address")
    if corrections:
        st.markdown('<div class="sec-head">Auto-corrections applied</div>', unsafe_allow_html=True)
        tags = "".join(f'<span class="corr-tag">↻ {c}</span>' for c in corrections)
        st.markdown(tags, unsafe_allow_html=True)
        if corrected:
            st.markdown('<div class="sec-head" style="margin-top:.75rem">Corrected address</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="reason-box" style="font-style:italic">{corrected}</div>', unsafe_allow_html=True)

    # Reason
    st.markdown('<div class="sec-head">Reason</div>', unsafe_allow_html=True)
    reason = result.get("reason", "—")
    st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

    # Rules triggered
    rules = result.get("rules_triggered") or []
    if rules:
        st.markdown(f'<div class="sec-head">Rules triggered ({len(rules)})</div>', unsafe_allow_html=True)
        for r in rules:
            name    = r.get("rule", "").replace("rule_", "").replace("_", " ")
            rreason = r.get("reason", "")
            pen     = r.get("penalty", 0.0)
            st.markdown(f"""
            <div class="rule-item">
              <span class="rule-name">{name}</span>
              <span class="rule-reason">{rreason}</span>
              <span class="rule-penalty">-{pen*100:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # Warnings
    warnings = result.get("warnings") or []
    if warnings:
        st.markdown('<div class="sec-head">Warnings</div>', unsafe_allow_html=True)
        tags = "".join(f'<span class="corr-tag" style="background:#faeeda;color:#854f0b">{w}</span>' for w in warnings)
        st.markdown(tags, unsafe_allow_html=True)

    # Suggestions
    suggestions = result.get("suggestions") or []
    if suggestions:
        st.markdown('<div class="sec-head">Suggested corrections</div>', unsafe_allow_html=True)
        for s in suggestions:
            stype = (s.get("type") or "").replace("_", " ")
            saddr = s.get("suggested_address") or ""
            st.markdown(f"""
            <div class="suggest-card">
              <div class="suggest-type">{stype}</div>
              <div class="suggest-addr">{saddr}</div>
            </div>
            """, unsafe_allow_html=True)

    # See Details expander (mirrors original app behaviour)
    with st.expander("🔎 See Details"):
        st.json(result)


# ─────────────────────────────────────────────────────────────────────────────
# MASTHEAD
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="masthead-eyebrow">India Post · ML + Rules</div>
<div class="masthead-title">Address <em>Validator</em></div>
<div class="masthead-sub">XGBoost model · pincode DB · 14-rule engine</div>
<hr class="masthead-divider">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_single, tab_batch = st.tabs(["Single address", "Batch validation"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ADDRESS
# ═══════════════════════════════════════════════════════════════════════════

with tab_single:

    st.markdown("Enter the shipping details below for validation.")

    # Helper for auto-fill
    def on_pincode_change():
        p = st.session_state.pincode.strip()
        if len(p) == 6 and p.isdigit():
            states = lookup_pincode(p)
            districts = lookup_pincode_districts(p)
            if states:
                st.session_state.state = states[0]
            if districts:
                options = sorted([d.title() for d in districts])
                st.session_state.city_options = options
                st.session_state.city = options[0]
            else:
                st.session_state.city_options = []

    # Multi-field Input Form
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("Address <span style='color:#ff4b4b'>*</span>", unsafe_allow_html=True)
        addr1 = st.text_area("Address*", placeholder="Shipping Address", key="addr1", height=100, label_visibility="collapsed")
    with row1_col2:
        st.markdown("Address 2 (Optional)", unsafe_allow_html=True)
        addr2 = st.text_area("Address 2 (Optional)", placeholder="Shipping Address 2", key="addr2", height=100, label_visibility="collapsed")

    st.markdown("Pincode <span style='color:#ff4b4b'>*</span>", unsafe_allow_html=True)
    pincode = st.text_input("Pincode*", placeholder="Enter pincode", key="pincode", label_visibility="collapsed", on_change=on_pincode_change)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.markdown("City <span style='color:#ff4b4b'>*</span>", unsafe_allow_html=True)
        city_opts = st.session_state.get("city_options", [])
        if len(city_opts) > 1:
            city = st.selectbox("City*", options=city_opts, key="city", label_visibility="collapsed")
        else:
            city = st.text_input("City*", placeholder="Enter city", key="city", label_visibility="collapsed")
            
    with row3_col2:
        st.markdown("State <span style='color:#ff4b4b'>*</span>", unsafe_allow_html=True)
        state = st.text_input("State*", placeholder="Enter state", key="state", label_visibility="collapsed")

    # Combine for prediction
    address_parts = [addr1, addr2, city, state]
    address = ", ".join([p for p in address_parts if p and p.strip()])
    if pincode:
        address += f" {pincode}"

    char_count = len(address) if address else 0
    st.caption(f"{char_count} chars")

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        validate_clicked = st.button(
            "validate address",
            type="primary",
            use_container_width=True,
            disabled=not bool(addr1.strip() and pincode.strip() and city.strip() and state.strip()),
        )
    with col_clear:
        if st.button("clear", use_container_width=True):
            for k in ["addr1", "addr2", "pincode", "city", "state"]:
                st.session_state[k] = ""
            st.session_state["city_options"] = []
            st.rerun()

    if validate_clicked:
        if not address.strip():
            st.warning("Please enter an address before validating.")
        else:
            with st.spinner("Analysing…"):
                try:
                    t0 = time.perf_counter()
                    result = predict(address)
                    elapsed = time.perf_counter() - t0
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            st.caption(f"Completed in {elapsed*1000:.0f} ms")
            render_verdict(result)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.caption("Enter one address per line (max 100 addresses)")

    batch_text = st.text_area(
        "Addresses (one per line)",
        height=200,
        placeholder="Flat 204, Sunrise Apartments, MG Road, Bangalore, Karnataka 560001\nPlot No 78, Sector 15, Chandigarh 160015\nasdfgh jklmnop",
        label_visibility="collapsed",
    )

    if st.button("validate batch", type="primary", disabled=not bool(batch_text and batch_text.strip())):
        raw_lines = [ln.strip() for ln in batch_text.splitlines() if ln.strip()]

        if not raw_lines:
            st.warning("No addresses found.")
        elif len(raw_lines) > 100:
            st.error("Maximum 100 addresses per batch. Please reduce your input.")
        else:
            with st.spinner(f"Validating {len(raw_lines)} addresses…"):
                try:
                    t0 = time.perf_counter()
                    results = predict_batch(raw_lines)
                    elapsed = time.perf_counter() - t0
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                    st.stop()

            valid_count   = sum(1 for r in results if r.get("prediction") == "VALID")
            invalid_count = len(results) - valid_count

            st.caption(f"{len(results)} addresses · {valid_count} valid · {invalid_count} invalid · {elapsed*1000:.0f} ms")

            # Summary metric row
            c1, c2, c3 = st.columns(3)
            c1.metric("Total",   len(results))
            c2.metric("Valid",   valid_count,   delta=None)
            c3.metric("Invalid", invalid_count, delta=None)

            # Results table
            import pandas as pd
            rows = []
            for i, (addr, res) in enumerate(zip(raw_lines, results), 1):
                rows.append({
                    "#":          i,
                    "Address":    addr[:80] + ("…" if len(addr) > 80 else ""),
                    "Prediction": res.get("prediction", "—"),
                    "Score":      f"{res.get('valid_score', 0)*100:.1f}%",
                    "Confidence": res.get("confidence", "—"),
                    "Reason":     (res.get("reason") or "")[:80],
                })

            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Prediction": st.column_config.TextColumn("Prediction", width="small"),
                    "Score":      st.column_config.TextColumn("Score",      width="small"),
                },
            )

            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                label="download CSV",
                data=csv,
                file_name="validation_results.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="font-size:11px;color:#8a8a85;display:flex;justify-content:space-between">'
    '<span>Address Validator v2.0</span>'
    '<span>'
    '<span class="badge">XGBoost</span>'
    '<span class="badge">India Post DB</span>'
    '<span class="badge">14 rules</span>'
    '</span></div>',
    unsafe_allow_html=True,
)