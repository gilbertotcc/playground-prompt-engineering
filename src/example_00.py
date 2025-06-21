from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool

# Load environment variables from .env file
load_dotenv()

shell_tool = ShellTool()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

examples = [{
    "Input": "I love this movie! It's fantastic.",
    "Sentiment": "Positive"
},
{
    "Input": "This is the worst experience I have ever had.",
    "Sentiment": "Negative"
},
{
    "Input": "The movie was okay, not the best but not the worst",
    "Sentiment": "Neutral"
}]

prompt_example = PromptTemplate(input_variables=["Input", "Sentiment"], template="Prompt: {Input}\n{Sentiment}")

prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = prompt_example,
    suffix = "Prompt: {Input}",
    input_variables = ["Input"]
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

multi_shot_prompt = prompt.format(Input="Despite the occasional setbacks and challenges, the team’s progress has been remarkable. There were moments of frustration, but the overall journey has been incredibly rewarding. It’s not always easy, but the sense of accomplishment makes it all worthwhile.")

print("--------------Response from the model--------------------")
print(chain.invoke(multi_shot_prompt))
