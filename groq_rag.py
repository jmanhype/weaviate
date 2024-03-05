import os
import uuid
import dspy
from dsp import LM
from groq import Groq as GroqClient  # Ensure this matches your actual Groq client import
from zep_python import ZepClient
from zep_python.user import CreateUserRequest
from zep_python.message import Message
from zep_python.memory import Session
from dspygen.utils.dspy_tools import init_dspy
from dspy import OpenAI, settings
from dspy.retrieve.weaviate_rm import WeaviateRM
from dspy.retrieve.retrieve import Retrieve
import weaviate
# Initialize GPT-4 with your OpenAI API key
gpt4 = dspy.OpenAI(model="gpt-4", max_tokens=2000, model_type="chat", api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Weaviate instance
weaviate_client = weaviate.Client("http://localhost:8080")

# Configure retrieval model to use 'ProductInfo' in Weaviate
retriever_model = WeaviateRM(weaviate_client=weaviate_client, weaviate_collection_name="ProductInfo", weaviate_collection_text_key="description")

# Configure DSPy settings with initialized models
settings.configure(lm=gpt4, rm=retriever_model)

# Check for necessary environment variables
assert os.environ.get("GROQ_API_KEY"), "GROQ_API_KEY environment variable is not set."
assert os.environ.get("ZEP_API_KEY"), "ZEP_API_KEY environment variable is not set."

class Groq(LM):
    def __init__(self, model="mixtral-8x7b-32768", **kwargs):
        super().__init__(model)  # Initialize the superclass with the model
        self.model = model  # Store the model as an instance attribute
        self.client = GroqClient(api_key=os.environ.get("GROQ_API_KEY"))
    
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        # Use the Groq API client to generate a response based on the prompt
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=kwargs.get("model", self.model),
        )
        # Ensure the API response is not empty or null
        assert chat_completion.choices and chat_completion.choices[0].message.content, "API response is empty or null."
        return [chat_completion.choices[0].message.content]
    
    def basic_request(self, prompt, **kwargs):
        # Placeholder implementation for basic requests
        return self.__call__(prompt, **kwargs)

class ZepIntegration:
    def __init__(self):
        self.client = ZepClient(base_url="http://localhost:8000", api_key=os.environ["ZEP_API_KEY"])


    def create_user_and_session(self, user_id, email, first_name, last_name, metadata):
        user_request = CreateUserRequest(
            user_id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            metadata=metadata,
        )
        self.client.user.add(user_request)
        session_id = uuid.uuid4().hex
        session = Session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        self.client.memory.add_session(session)
        return session_id

    def add_chat_message(self, session_id, messages):
        for message in messages:
            msg = Message(role=message['role'], content=message['content'])
            self.client.memory.add_message(session_id=session_id, message=msg)

    def get_sessions_for_user(self, user_id):
        return self.client.user.getSessions(user_id)

class ProductDetails:
    def __init__(self, title="Title not available", description="Description not available", price="Price not available", stock_status="Stock status not available", link="Link not available", image="Image not available"):
        self.title = title
        self.description = description
        self.price = price
        self.stock_status = stock_status
        self.link = link
        self.image = image


import weaviate

class DetailedProductInfoRetriever(dspy.Module):
    def __init__(self, k=3, weaviate_client=None):
        super().__init__()
        self.k = k  # Number of results to retrieve
        # Use an existing Weaviate client or initialize a new one
        self.weaviate_client = weaviate_client or weaviate.Client("http://localhost:8080")

    def forward(self, query):
        print(f"Executing query: {query}")
        
        # Construct the GraphQL query as a string
        graphql_query = f"""
        {{
            Get {{
                ProductInfo(where: {{operator: Like, path: ["description"], valueString: "%{query}%"}} limit: {self.k}) {{
                    title
                    description
                    price
                }}
            }}
        }}
        """

        # Execute the query using the raw method, which expects a string
        try:
            response = self.weaviate_client.query.raw(graphql_query)
        except Exception as e:
            print(f"An error occurred while querying Weaviate: {e}")
            return ProductDetails()

        # Check if the query returned any products
        if response and 'data' in response and 'Get' in response['data'] and 'ProductInfo' in response['data']['Get'] and len(response['data']['Get']['ProductInfo']) > 0:
            product = response['data']['Get']['ProductInfo'][0]  # Using the first product for demonstration

            # Populate ProductDetails with the retrieved information
            return ProductDetails(
                title=product.get('title', 'Title not available'),
                description=product.get('description', 'Description not available'),
                price=str(product.get('price', 'Price not available'))  # Convert price to string for printing
            )
        else:
            print("No matching product found.")
            return ProductDetails()

# Adjust chat_loop() or similar where this class is instantiated
# Ensure weaviate_client is passed to DetailedProductInfoRetriever





class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Often between 1 and 5 words")

class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="May contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Often between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.history = []
        self.zep_integration = ZepIntegration()

    def forward(self, question):
        self.history.append(question)
        retrieved_context = self.retrieve(question).passages
        context = "\n".join(retrieved_context)
        combined_context = "\n".join(self.history) + "\n" + context
        prediction = self.generate_answer(context=combined_context, question=question)
        self.history.append(prediction.answer)
        return {"context": context, "answer": prediction.answer}

# Main function to initialize and run the chatbot
def chat_loop():
    print("Welcome to the product information chatbot. Type 'quit' to exit.")
    retriever = DetailedProductInfoRetriever(k=3)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        product_details = retriever(user_input)
        # Display the product details
        print(f"Product Details:\nTitle: {product_details.title}\nDescription: {product_details.description}\nPrice: {product_details.price}\nStock Status: {product_details.stock_status}\nLink: {product_details.link}\nImage: {product_details.image}")

if __name__ == "__main__":
    chat_loop()
