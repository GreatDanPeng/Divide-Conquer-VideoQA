from openai import OpenAI
client = OpenAI(
    api_key = "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
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo-0125",
)