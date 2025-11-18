from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv() 

class Generator:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_kEY")
        
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        
generator = Generator()