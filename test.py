import ollama

response = ollama.chat(
    model="gemma4:e4b",
    messages=[
        {
            "role": "user",
            "content": "今天是几月几号",
        },
    ],
)

print(response["message"]["content"])
