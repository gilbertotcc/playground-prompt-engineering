from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool

# from langchain_ollama.llms import OllamaLLM

# Load environment variables from .env file
load_dotenv()

shell_tool = ShellTool()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

# llm = OllamaLLM(model="deepseek-r1")

prompt = """ Q: Mrs. Smith currently has 10 students in her class.
She receives 3 new groups of students over the week. Each group contains 4 students.
How many students does Mrs. Smith have in her class now?
A: Let's think step by step
"""

messages = [SystemMessage(content=prompt), HumanMessage(content=prompt)]

response = llm.invoke(messages)

output = response.content
metadata = response.usage_metadata

print("--------------Response from the model--------------------")
print(output)

print("--------------Response from the model--------------------")
print(metadata)