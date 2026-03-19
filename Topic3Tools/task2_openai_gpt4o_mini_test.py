"""
Task 2: OpenAI GPT-4o-mini Test

Verifies OPENAI_API_KEY is set and GPT-4o-mini responds correctly.
Set key: export OPENAI_API_KEY="your-actual-key"
Or in ~/.profile: export OPENAI_API_KEY="your-actual-key"
"""

import os
from openai import OpenAI

# client = OpenAI() uses api_key=os.getenv("OPENAI_API_KEY") by default
# If the key is not set, OpenAI() will raise an error when making requests
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say: Working!"}],
    max_tokens=5
)

print(f"Success! Response: {response.choices[0].message.content}")
if response.usage:
    cost = response.usage.total_tokens * 0.000000375  # approximate
    print(f"Cost: ${cost:.6f}")
