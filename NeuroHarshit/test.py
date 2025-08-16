from openai import OpenAI
from dotenv import load_dotenv
from os import getenv


load_dotenv()
api_key = getenv('API_KEY')

client = OpenAI(api_key= api_key)

response = client.responses.create(
    model="gpt-4o-mini",
    input="Explain in detail, in easy language, use bullet points, word limit is 500 words: `What is attention in transformer?`"
)

print(response.output_text)