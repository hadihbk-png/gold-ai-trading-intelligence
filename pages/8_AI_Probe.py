"""
TEMPORARY diagnostic v2 — isolate the AI-brief failure.
Runs three Anthropic calls to find exactly which parameter breaks it.
Read-only. DELETE after reading the result.
"""
import streamlit as st

st.set_page_config(page_title="AI Probe", page_icon="🩺", layout="wide")
st.title("🩺 Anthropic API probe v2 (temporary)")
st.caption("Finds exactly which call parameter breaks the brief. Delete after use.")


def _secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


key = _secret("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"
SYS = (
    "You are a professional gold market analyst writing a short daily brief "
    "for an investor. Keep it under 50 words. Research context only, not advice."
)

st.write(f"key present: **{bool(key)}** · length **{len(key)}** · starts `sk-ant-`: "
         f"**{key.startswith('sk-ant-') if key else False}** · model **{MODEL}**")


def run(label: str, **kwargs):
    st.markdown(f"**{label}**")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(model=MODEL, **kwargs)
        txt = resp.content[0].text[:120] if resp.content else "(empty)"
        st.success(f"✅ OK — {txt}")
    except Exception as e:
        st.error(f"❌ {type(e).__name__}")
        st.code(str(e)[:900])


if not key:
    st.error("No ANTHROPIC_API_KEY in secrets.")
else:
    # A — simple call (already known to work)
    run("A. Simple call (no system, no cache)",
        max_tokens=20, messages=[{"role": "user", "content": "Reply with: OK"}])

    # B — EXACTLY what the app does: system list WITH cache_control ephemeral
    run("B. system list + cache_control:ephemeral  (THE APP'S CALL)",
        max_tokens=200, temperature=0.3,
        system=[{"type": "text", "text": SYS, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "Write a two-line gold brief."}])

    # C — same but WITHOUT cache_control (candidate fix)
    run("C. plain system string, NO cache_control  (CANDIDATE FIX)",
        max_tokens=200, temperature=0.3,
        system=SYS,
        messages=[{"role": "user", "content": "Write a two-line gold brief."}])

    st.info("Diagnosis: if **B fails** and **A & C succeed**, the bug is the "
            "`cache_control` on the short system prompt — the fix is to remove it. "
            "Send all three results to Claude.")
