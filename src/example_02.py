from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.vectorstores import InMemoryVectorStore

file_path = "./development/businesses_document.pdf"
loader = PyPDFLoader(file_path, mode="single")

docs = loader.load()

print("===== Document content =====")
document_content = docs[0].page_content
print(document_content)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-2.0-flash")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(docs)
