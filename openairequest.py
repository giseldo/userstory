from openai import OpenAI

client = OpenAI()

text = 'As a UI designer, I want to redesign the Resources page, so that it matches the new Broker design styles'

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a scrum master, skilled in create better user story for agile software projects."},
        {"role": "user", "content":"How can i improve this this user story : {}".format(text)}
    ]
)

print(completion.choices[0].message)