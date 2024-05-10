import requests

API_URL = "https://api-inference.huggingface.co/models/Hashif/legalLlama-2-7b-chat-finetune"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})
