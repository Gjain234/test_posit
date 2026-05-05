import json

from flask import Flask, render_template_string, request
from itsai.platform.authentication import DesktopToken
from openai import AzureOpenAI


DEFAULT_PROMPT = "Explain the importance of financial inclusion."
DEFAULT_SYSTEM_PROMPT = "You are an AI assistant helping with World Bank projects."
DEFAULT_ENDPOINT = "https://azapimdev.worldbank.org/conversationalai/v2/"
DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_MODEL = "gpt-5-mini"

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>itsai Posit Credential Test</title>
    <style>
        :root {
            color-scheme: light;
            font-family: Arial, sans-serif;
        }
        body {
            margin: 0;
            background: #f5f7fb;
            color: #162033;
        }
        main {
            max-width: 880px;
            margin: 40px auto;
            padding: 32px;
            background: #ffffff;
            border: 1px solid #d8dfeb;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(15, 30, 60, 0.08);
        }
        h1 {
            margin-top: 0;
            font-size: 28px;
        }
        p {
            line-height: 1.5;
        }
        form {
            display: grid;
            gap: 16px;
            margin-top: 24px;
        }
        label {
            display: grid;
            gap: 6px;
            font-weight: 600;
        }
        input, select, textarea, button {
            font: inherit;
        }
        input, select, textarea {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #b8c4d9;
            border-radius: 8px;
            box-sizing: border-box;
            background: #ffffff;
        }
        textarea {
            min-height: 120px;
            resize: vertical;
        }
        .grid {
            display: grid;
            gap: 16px;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }
        button {
            width: fit-content;
            padding: 12px 18px;
            border: 0;
            border-radius: 8px;
            background: #0c63e7;
            color: #ffffff;
            cursor: pointer;
            font-weight: 600;
        }
        .card {
            margin-top: 28px;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #d8dfeb;
            background: #fbfcfe;
        }
        pre {
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
            background: #0f172a;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <main>
        <h1>itsai Posit Credential Test</h1>
        <p>Use this app to verify whether itsai token acquisition and the Azure OpenAI call both work in your Posit deployment.</p>

        <form method="post">
            <div class="grid">
                <label>
                    itsai environment
                    <select name="environment">
                        {% for option in ["DEV", "UAT", "PROD"] %}
                        <option value="{{ option }}" {% if values.environment == option %}selected{% endif %}>{{ option }}</option>
                        {% endfor %}
                    </select>
                </label>
                <label>
                    Model
                    <input name="model" value="{{ values.model }}">
                </label>
                <label>
                    API version
                    <input name="api_version" value="{{ values.api_version }}">
                </label>
                <label>
                    Max completion tokens
                    <input name="max_completion_tokens" type="number" min="1" max="4000" value="{{ values.max_completion_tokens }}">
                </label>
            </div>

            <label>
                Azure endpoint
                <input name="endpoint" value="{{ values.endpoint }}">
            </label>

            <label>
                System prompt
                <textarea name="system_prompt">{{ values.system_prompt }}</textarea>
            </label>

            <label>
                User prompt
                <textarea name="user_prompt">{{ values.user_prompt }}</textarea>
            </label>

            <button type="submit">Run credential test</button>
        </form>

        {% if response_text is not none %}
        <section class="card">
            <h2>Response</h2>
            <p>{{ response_text }}</p>
        </section>
        <section class="card">
            <h2>Raw response</h2>
            <pre>{{ response_json }}</pre>
        </section>
        {% endif %}
    </main>
</body>
</html>
"""


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


app = Flask(__name__)


def default_values():
    return {
        "environment": "DEV",
        "endpoint": DEFAULT_ENDPOINT,
        "api_version": DEFAULT_API_VERSION,
        "model": DEFAULT_MODEL,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_PROMPT,
        "max_completion_tokens": "2000",
    }


@app.route("/", methods=["GET", "POST"])
def home():
    values = default_values()
    response_text = None
    response_json = None

    if request.method == "POST":
        values.update(
            {
                "environment": request.form["environment"],
                "endpoint": request.form["endpoint"],
                "api_version": request.form["api_version"],
                "model": request.form["model"],
                "system_prompt": request.form["system_prompt"],
                "user_prompt": request.form["user_prompt"],
                "max_completion_tokens": request.form["max_completion_tokens"],
            }
        )
        response = run_completion(
            environment=values["environment"],
            endpoint=values["endpoint"],
            api_version=values["api_version"],
            model=values["model"],
            system_prompt=values["system_prompt"],
            user_prompt=values["user_prompt"],
            max_completion_tokens=int(values["max_completion_tokens"]),
        )
        response_text = response.choices[0].message.content if response.choices else ""
        response_json = json.dumps(response.model_dump(), indent=2)

    return render_template_string(
        PAGE_TEMPLATE,
        values=values,
        response_text=response_text,
        response_json=response_json,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)