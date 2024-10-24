import requests
import json
import os

# Specify a URL that resolves to your workspace
BASE_URL = "http://127.0.0.1:8000"

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_model_path = os.path.join(config['output_model_path'])

# Define a helper function to call an API and get the response as JSON
def get_api_response(endpoint):
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to get response from {url}, status code: {response.status_code}"}

# Call each API endpoint and store the responses
response1 = get_api_response('/prediction?filename=testdata/testdata.csv')
response2 = get_api_response('/scoring')
response3 = get_api_response('/summarystats')
response4 = get_api_response('/diagnostics')

# Combine all API responses into a dictionary
combined_responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4
}

# Write the responses to a file in a pretty-printed JSON format
output_file_path = os.path.join(output_model_path, 'apireturns.txt')
with open(output_file_path, 'w') as f:
    json.dump(combined_responses, f, indent=4)

print(f"Responses saved to {output_file_path}")