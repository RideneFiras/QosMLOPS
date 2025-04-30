import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)


def generate_qos_insight(throughput, shap_values, features):
    formatted_features = "\n".join(
        [
            f"- {feat}: {'+' if val >= 0 else ''}{round(val, 3)} Mbps"
            for feat, val in zip(features, shap_values)
        ]
    )

    prompt = f"""
You are a telecom and AI expert assisting a 5G network optimization platform.

The system has predicted a user's download throughput: **{round(throughput, 2)} Mbps**, based on detailed network and environmental measurements.

Below are the key factors that influenced the user's throughput, showing how each one increased or decreased it (in Mbps):

{formatted_features}

---

Based on this, provide a short analysis of the user's **Quality of Service (QoS)** experience.

Your response must include:

1. **QoS Rating**
2. **Main Factors Affecting Throughput**
3. **Recommendations**

✅ Keep it professional, clear, and easy to understand.
❌ Do not include raw numbers — summarize the insights.
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
        # "stream": True,
    )

    return response.choices[0].message.content
