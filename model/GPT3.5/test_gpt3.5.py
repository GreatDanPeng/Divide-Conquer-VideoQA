from openai import OpenAI
client = OpenAI()
messages=[
    {"role": "system", "content": "You are a helpful assistant. You are helping the user summarize a document."},
    {
        "role": "user",
        "content": ""
    }
]
# Import a json file with segments of text
import json
with open('model/GPT3.5/202_results.json') as f:
    data = json.load(f)
    
# Write a function to extract the segments from the json file, the segments are stored in a list by the key from segment1 to the last segment
def extract_segments(data, messages):
    segments = []
    for i in range(1, int((len(data) - 5)/2)): # test half of the segments since it is a large file
        messages[0]['content'] += data['segment' + str(i)]
        segments.append(data['segment' + str(i)])
    return segments, messages

sum_segments, messages = extract_segments(data, messages)
messages[0]['content'] += "Can you summarize the document for me in 1000 words?"

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=messages
)

print(completion.choices[0].message.content)