import os
import dspy
from dsp import LM
from groq import Groq as GroqClient
from dspygen.utils.dspy_tools import init_dspy

# Ensure the GROQ_API_KEY environment variable is set
assert os.environ.get("GROQ_API_KEY"), "GROQ_API_KEY environment variable is not set."

# Custom Groq Language Model
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

# BasicQA Signature for simple question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# GenerateAnswer Signature for Retrieval-Augmented Generation
class GenerateAnswer(dspy.Signature):
    """Generate answers based on context and questions."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# RAG Module for context-aware response generation
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)  # Set up retrieval model
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)  # Set up response generation
        self.history = []  # Initialize conversation history

    def forward(self, question):
        # Append the latest question to history
        self.history.append(question)
        
        # Retrieve context based on the latest question
        retrieved_context = self.retrieve(question).passages
        context = "\n".join(retrieved_context)
        
        # Combine retrieved context with conversation history for richer context
        combined_context = "\n".join(self.history) + "\n" + context
        
        # Generate an answer based on the combined context
        prediction = self.generate_answer(context=combined_context, question=question)
        
        # Append the generated answer to history
        self.history.append(prediction.answer)
        
        return {"context": context, "answer": prediction.answer}

# Main function to initialize and run the chatbot
def main():
    init_dspy(max_tokens=2000)
    
    # Configure DSPy with custom Groq model and ColBERTv2 retrieval model
    groq_model = Groq(model="llama2-70b-4096")
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
    dspy.settings.configure(lm=groq_model, rm=colbertv2_wiki17_abstracts)
    
    # Initialize the RAG module for conversation
    rag = RAG(num_passages=3)
    
    print("Welcome to the Chatbot. Type 'quit' to exit.")
    
    # Chat loop for continuous interaction
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        result = rag(user_input)
        print("Bot:", result['answer'])

if __name__ == '__main__':
    main()
