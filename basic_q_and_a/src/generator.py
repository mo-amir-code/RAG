import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

class OpenAILLM:
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize OpenAI LLM
        Args:
            api_key: OpenAI API key to use their model
            model_name: Open AI model name
        """
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        print(f"[DEBUG] OPENAI API KEY: {self.api_key}")
        
        if not self.api_key:
            raise ValueError("[ERROR] OPENAI API key is required. Set OPENAI_API_KEY environment variable or pass api key parameter")

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=1024
        )
        
        print(f"[DEBUG] Initialized OPENAI model with model:  {self.model_name}")
        
        
        
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using retrieved context
        Args:
            query: User questions
            context: Retrieved documents context

        Returns: 
            Generated response in string
        """
        
        # Create promt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

                Context:
                {context}

                Question: {question}

                Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        )   
        
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            # print(f"[DEBUG] RESPONSE inside OPENAI_LLM class: {response}")
            return response.content
        except Exception as e:
            return f"[ERROR] generating response: {str(e)}"
               
        
openai_llm = OpenAILLM()

if __name__ == "__main__":
    openai_llm = OpenAILLM()
    openai_llm.generate_response("give any thing", "no context")
