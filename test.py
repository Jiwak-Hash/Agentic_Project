from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5-coder:latest",
    temperature=0
)

response = llm.invoke(
    "What is the intent of: Where is my order?"
)

print(response.content)