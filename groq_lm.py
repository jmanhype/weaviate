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
