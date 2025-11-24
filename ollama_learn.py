import requests
prompt = "Hello"

def ollama_running():
    try:
        response = requests.get("http://localhost:11434",timeout=5)
        return response.status_code == 200
    except:
        return False

if not ollama_running():
    print("Ollama not running")
else:
    print("ollama running as expected")
                                

response = requests.post("http://localhost:11434/api/chat",
                          json={
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False})

data = response.json()
print(data["messages"]["content"])

def basic_prompt(sys_prompt,usr_prompt):
    response = requests.post("http://localhost:11434/api/chat",
                             json={
                                 "model": "llama3.1:8b",
                                 "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":"usr_prompt"}],
                                 "stream":False
                             })
    if response.status_code == 200:
        result = response.json()
        print(result["message"]["content"])
    else:
        print(f"Effor : {response.status_code}")

print("Basic Examples:-")
sys_prompt = "You are helpful assistant"
usr_prompt = "What is python?"

basic_prompt(sys_prompt,usr_prompt)

