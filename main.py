import os

API_Token = os.environ['API_Token']

from huggingface_hub.inference_api import InferenceApi

inference = InferenceApi(repo_id="gpt2-large", token=API_Token)

# define our parameters
top_p_params = {"max_length": 128, "top_p": 0.95, "do_sample":True} # top-p (nucleus sampling)
top_k_params = {"max_length": 128,  "top_k": 4, "do_sample":True} # top-k
contrastive_params = {"max_length": 128, "penalty_alpha": 0.6, "top_k": 4}

input_string = "Learn about AI because"
result = inference(input_string)
print(result)

result = inference(input_string, top_p_params)
print("\nTop P: {}".format(result))

result = inference(input_string, top_k_params)
print("\nTop K: {}".format(result))

result = inference(input_string, contrastive_params)
print("\nContrastive: {}".format(result))