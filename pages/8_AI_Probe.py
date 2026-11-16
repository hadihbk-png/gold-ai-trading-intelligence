"""
TEMPORARY diagnostic — Anthropic API probe.

The AI brief/assistant catches every error and returns None, so the real
cause is hidden. This page makes one live call and shows the EXACT error.
Read-only. DELETE this page after we've read the result.
"""
import streamlit as st

st.set_page_config(page_title="AI Probe", page_icon="🩺", layout="wide")
st.title("🩺 Anthropic API probe (temporary)")
st.caption("Surfaces why the AI brief/assistant fails on Cloud. Delete after use.")


def _secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


key = _secret("ANTHROPIC_API_KEY")
MODEL = "claude-sonnet-4-6"

st.subheader("Key & config (no secret values shown)")
st.write(f"- `ANTHROPIC_API_KEY` present: **{bool(key)}**")
st.write(f"- key length: **{len(key)}** characters")
st.write(f"- starts with `sk-ant-`: **{key.startswith('sk-ant-') if key else False}**")
st.write(f"- model in code: **{MODEL}**")

st.subheader("Live test call")
if not key:
    st.error("No ANTHROPIC_API_KEY found in secrets.")
else:
    try:
        import anthropic
        st.write(f"- anthropic SDK version: **{anthropic.__version__}**")
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        text = resp.content[0].text if resp.content else "(empty)"
        st.success(f"✅ SUCCESS — the model replied: {text}")
        st.info("If this succeeds, the key + model work — the brief failure is elsewhere.")
    except Exception as e:
        st.error(f"❌ FAILED — {type(e).__name__}")
        st.code(str(e)[:1000])
        st.info(
            "What the error means:\n"
            "- AuthenticationError / 401 → the key value is wrong/expired\n"
            "- 'credit balance is too low' / 400 → add billing credits in the Anthropic Console\n"
            "- NotFoundError / 404 model → model not available to this key\n"
            "- RateLimitError / 429 → rate limited, retry\n"
            "Send this whole screen to Claude."
        )
