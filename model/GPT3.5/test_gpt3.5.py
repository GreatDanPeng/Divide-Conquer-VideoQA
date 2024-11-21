from openai import OpenAI
client = OpenAI()

text1 = 'The video shows a woman doing yoga on a rooftop in Spain. She is dressed in black pants, black leggings, and a black tank top.  There are mountains and hills in the background, and a blue sky is visible.  The woman is doing various yoga poses, such as downward facing dog, plank, and mountain pose, on the tiled roof.  She is focused and determined, and the peaceful atmosphere of the place enhances her state of mind.  The video is a great example of how yoga can be practiced in a beautiful outdoor setting.'
text2 = 'The video shows a young woman doing yoga outdoors on a rooftop. She stands with her arms stretched out in the shape of a large star and a small triangle.  A mountain is visible in the distance.  The video shows a number of yoga poses, including the plank and the warrior pose.  The woman is wearing a black sports bra and gray leggings.  She is also wearing a black cap and black and white shoes.  In one scene, she performs a move called downward facing dog.  The video ends with her practicing another yoga pose, the triangle pose, with her hands behind her head.  The woman is fit and strong, with a lean and toned physique.  The video is a fun and interesting introduction to the world of yoga and shows the benefits of practicing yoga outdoors.'
question = 'These two pieces of text come from two adjacent clips in a video, please merge these two pieces of text without missing details.'
messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": ""
    }
]
messages[1]['content'] = text1 + '\n' + text2 + '\n' + question

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=messages
)

print(completion.choices[0].message.content)