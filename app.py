import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class FitnessCoachApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Fitness_Guide.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Load and process PDF file into documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Build vector database for document retrieval."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant documents based on query."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = FitnessCoachApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    """Generate a response based on user message and historical context."""
    system_message = ("You are a knowledgeable fitness coach chatbot. You provide accurate and helpful advice on various fitness topics. "
                      "This includes workout routines, nutrition, exercise techniques, and tips for maintaining a healthy lifestyle.")
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for msg in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = msg.choices[0].delta.content
        response += token
        yield response

# Define the Gradio app interface
demo = gr.Blocks()

with demo:
    gr.Markdown("💪 **Fitness Coach Chatbot**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on fitness guidebooks that are publicly available. "
        "We are not certified fitness professionals, and the use of this chatbot is at your own risk. Consult a professional for personalized advice.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["What are some effective workout routines for beginners?"],
            ["How can I improve my diet for better fitness results?"],
            ["Can you suggest exercises for building muscle?"],
            ["What are the benefits of regular cardio exercise?"],
            ["How can I stay motivated to exercise regularly?"],
            ["What are some tips for recovering from a workout?"],
            ["How can I balance strength training and cardio?"],
            ["What are the common mistakes in weightlifting and how to avoid them?"]
        ],
        title='Fitness Coach Chatbot 💪'
    )

if __name__ == "__main__":
    demo.launch()

