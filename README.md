# Fitness Coach RAG Chatbot

Welcome to the Fitness Coach RAG Chatbot project! This application provides personalized fitness advice, workout routines, and progress tracking by combining document retrieval with a language model.

## Features

- **Personalized Fitness Advice:** Get tailored advice on fitness routines and nutrition.
- **Workout Routines:** Generate custom workout plans based on your goals.
- **Progress Tracking:** Track your fitness progress.
- **Document Retrieval:** Uses a fitness guide to enhance responses with specific information.

## Setup

### Prerequisites

Ensure you have Python 3.7+ installed on your system. You will need to install the packages listed in `requirements.txt`.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/fitness-coach-rag-chatbot.git
    cd fitness-coach-rag-chatbot
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download and place your fitness-related PDF as `Fitness_Guide.pdf` in the same directory as `app.py`.

### Running the App

To start the application, run:

```bash
python app.py


Building a Retrieval-Augmented Generation (RAG) bot can significantly enhance the capabilities of a language model by incorporating external knowledge to generate more accurate and contextually relevant responses. This guide will walk you through creating a simple RAG bot using Gradio and the Hugging Face APIs.

But how does RAG enhance LLM’s performance?

RAG improves the performance of language models by augmenting them with external documents. This method retrieves relevant documents based on the user query and combines them with the original prompt before passing them to the language model for response generation. This approach ensures that the language model can access up-to-date and domain-specific information without the need for extensive retraining.



A common scenario of RAG helping LLM (Source)

The basic steps in RAG can be simplified as follows:

Input: The question to which the LLM system responds is referred to as the input. If no RAG is used, the LLM is directly used to respond to the question.

Indexing: If RAG is used, then a series of related documents are indexed by chunking them first, generating embeddings of the chunks, and indexing them into a vector store. At inference, the query is also embedded in a similar way.


Basic retrieval steps in RAG. (Source)

Retrieval: The relevant documents are obtained by comparing the query against the indexed vectors, also denoted as “Relevant Documents”.

Generation: The relevant documents are combined with the original prompt as additional context. The combined text and prompt are then passed to the model for response generation which is then prepared as the final output of the system to the user.

In the example provided, using the model directly fails to respond to the question due to a lack of knowledge of current events. On the other hand, when using RAG, the system can pull the relevant information needed for the model to answer the question appropriately. (Source)

Now Let’s Build a Chatbot using RAG:

I have used Zephyr LLM model and all-MiniLM-L6-v2 sentence transformer model. This sentence-transformers model maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

The all-* models were trained on all available training data (more than 1 billion training pairs) and are designed as general purpose models. The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality. Toggle All models to see all evaluated original models.

We need the following ingredients:

1. A PDF as your knowledgebase

2. A requirements.txt file

3. An app.py file

4. An account on Hugging Face (See this blog to learn about building a LLM chatbot in Hugging Face)
