import falcon
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LanguageModelResource:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("Hashif/legalLlama-2-7b-chat-finetune")
        self.model = AutoModelForCausalLM.from_pretrained("Hashif/legalLlama-2-7b-chat-finetune").to(self.device)

    def generate_text(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(inputs.input_ids, max_length=30)
        generated_text = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return generated_text

    def on_post(self, req, resp):
        try:
            # Get input text from the request
            input_data = json.loads(req.bounded_stream.read().decode("utf-8"))
            input_text = input_data["text"]

            # Generate text using the language model
            generated_text = self.generate_text(input_text)

            # Return the generated text as response
            resp.body = json.dumps({"generated_text": generated_text})
            resp.status = falcon.HTTP_200
        except Exception as e:
            resp.body = json.dumps({"error": str(e)})
            resp.status = falcon.HTTP_500

# Create the Falcon application
app = falcon.App()

# Define the route for the language model
language_model_resource = LanguageModelResource()
app.add_route("/generate_text", language_model_resource)

if __name__ == "__main__":
    # Run the Falcon application
    import os
    port = int(os.environ.get("PORT", 8000))
    from wsgiref import simple_server
    httpd = simple_server.make_server("0.0.0.0", port, app)
    print(f"Server running on port {port}")
    httpd.serve_forever()

