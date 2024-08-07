# Gemini-powered-RAG-Chatbot
This project showcases an AI Chatbot designed to provide clear and informative answers to user questions by leveraging advanced AI models and text retrieval techniques. The chatbot utilizes a generative model from Google for creating responses based on the provided document data.

## How It Works

1. **Embeddings:**
   - **Embeddings:** The text chunks are converted into embeddings using the HuggingFaceEmbeddings model.
   - **Text Splitting:** The provided document is split into manageable chunks using the RecursiveCharacterTextSplitter.
   - **Vector db:** The embeddings and associated metadata are stored in Qdrant for efficient retrieval.

2. **Rerival:**
   * **Text Input:** The user enters a question.
   * **Retrieval and QA:** The user's question is processed to retrieve relevant information from the document storage, and the response 
       is generated using the Google Generative AI model.
   * **Response Generation:** The chatbot combines the retrieved information and generates a comprehensive response.

## Files and Directories
* **app.py:** The main script containing the code for the chatbot.
* **brookline_data.txt:** The document file that contains the data used by the chatbot for information retrieval.
* **README.md:** This file providing an overview and instructions for the project.
* **requirements.txt**
## Set Environment Variables:
**google_api_key:** Your Google API key for Google Generative AI.

## Usage
**Prepare the Document:** Ensure `brookline_data.txt` is in the project directory.
## Example
- **User:** "What information is available about the Brookline bank?"
- **Chatbot:** "retrieved data and generated response"
  
## Technology Stack
- Gradio
- Langchain
- HuggingFace
- Google Generative AI
- Qdrant
  
## [Hugging Face URL](https://huggingface.co/spaces/ahmadmac/Query-Chatbot)
