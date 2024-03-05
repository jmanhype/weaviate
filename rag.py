import os
import json
import dspy
from dspy import OpenAI, settings
from dspy.retrieve.weaviate_rm import WeaviateRM
from dspy.retrieve.retrieve import Retrieve

import weaviate

# Initialize GPT-4 with your OpenAI API key and desired settings
gpt4 = dspy.OpenAI(model="gpt-4", max_tokens=2000, model_type="chat", api_key="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")
print("GPT-4 model initialized.")

# Connect to your local Weaviate instance
weaviate_client = weaviate.Client("http://localhost:8080")
print("Connected to Weaviate.")

# Configure the retrieval model to use the 'ProductInfo' collection in your Weaviate instance
retriever_model = WeaviateRM(weaviate_collection_name="ProductInfo", weaviate_client=weaviate_client, weaviate_collection_text_key="description")
print("Retrieval model configured for 'ProductInfo' collection.")

# Update DSPy settings with the initialized models
settings.configure(lm=gpt4, rm=retriever_model)
print("DSPy settings updated.")

class ProductDetails:
    def __init__(self, title="Title not available", description="Description not available", price="Price not available", stock_status="Stock status not available", link="Link not available", image="Image not available"):
        self.title = title
        self.description = description
        self.price = price
        self.stock_status = stock_status
        self.link = link
        self.image = image

class DetailedProductInfoRetriever(dspy.Module):
    def __init__(self, k=3):
        super().__init__()
        self.retriever = Retrieve(k=k)

    def forward(self, query):
        print(f"Executing query: {query}")
        response = self.retriever(query)

        if response.passages:
            try:
                # Attempt to parse the first passage as JSON to extract detailed product information
                product_info = json.loads(response.passages[0])

                # Extract detailed product information
                return ProductDetails(
                    title=product_info.get("title", "Title not available"),
                    description=product_info.get("description", "Description not available"),
                    price=product_info.get("price", "Price not available"),
                    stock_status="In stock" if product_info.get("inStock", False) else "Out of stock",
                    link=product_info.get("link", "Link not available"),
                    image=product_info.get("image", "Image not available")
                )
            except json.JSONDecodeError:
                print("Failed to decode product info from the retrieved passage.")
                return ProductDetails(description="Failed to retrieve product information.")
        else:
            print("No matching product found.")
            return ProductDetails(description="No matching product found.")


# Chat loop for interactive conversation with the user
def chat_loop():
    print("Welcome to the product information chatbot. Type 'quit' to exit.")
    detailed_retriever = DetailedProductInfoRetriever(k=3)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        product_details = detailed_retriever(user_input)
        # Format and display the retrieved product details
        print(f"Product Details:\nTitle: {product_details.title}\nDescription: {product_details.description}\nPrice: {product_details.price}\nStock Status: {product_details.stock_status}\nLink: {product_details.link}\nImage: {product_details.image}")

if __name__ == "__main__":
    chat_loop()
