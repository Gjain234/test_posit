import streamlit as st
from itsai.platform.authentication import DesktopToken
from openai import AzureOpenAI


DEFAULT_PROMPT = "Explain the importance of financial inclusion."
DEFAULT_SYSTEM_PROMPT = "You are an AI assistant helping with World Bank projects."
DEFAULT_ENDPOINT = "https://azapimdev.worldbank.org/conversationalai/v2/"
DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_MODEL = "gpt-5-mini"


def create_client(environment: str, endpoint: str, api_version: str):
    token_class = DesktopToken()
    token_provider = lambda: token_class.token_provider(env=environment)
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )


def run_completion(
    environment: str,
    endpoint: str,
    api_version: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
):
    client = create_client(environment=environment, endpoint=endpoint, api_version=api_version)
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_completion_tokens,
        reasoning_effort="low",
    )


st.set_page_config(page_title="itsai Posit Credential Test", page_icon="AI", layout="centered")
st.title("itsai Posit Credential Test")
st.write(
    "Use this app to verify whether itsai token acquisition and the Azure OpenAI call both work in your Posit deployment."
)

with st.form("credential_test_form"):
    environment = st.selectbox("itsai environment", options=["DEV", "UAT", "PROD"], index=0)
    endpoint = st.text_input("Azure endpoint", value=DEFAULT_ENDPOINT)
    api_version = st.text_input("API version", value=DEFAULT_API_VERSION)
    model = st.text_input("Model", value=DEFAULT_MODEL)
    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=100)
    user_prompt = st.text_area("User prompt", value=DEFAULT_PROMPT, height=140)
    max_completion_tokens = st.number_input(
        "Max completion tokens", min_value=1, max_value=4000, value=2000, step=100
    )
    submitted = st.form_submit_button("Run credential test")

if submitted:
    with st.spinner("Requesting token and calling Azure OpenAI..."):
        response = run_completion(
            environment=environment,
            endpoint=endpoint,
            api_version=api_version,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_completion_tokens=int(max_completion_tokens),
        )

    content = response.choices[0].message.content if response.choices else ""
    st.success("Credential test succeeded.")
    st.subheader("Response")
    st.write(content or "No content returned.")
    with st.expander("Raw response"):
        st.json(response.model_dump())