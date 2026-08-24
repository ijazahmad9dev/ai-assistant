from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="https://own-particular-becomes-distinction.trycloudflare.com",
    model="gpt-oss:latest",
)

response = llm.invoke("Say hello in one sentence.")
print(response.content)