import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain_huggingface import HuggingFaceEndpoint,HuggingFacePipeline
import os
from transformers import pipeline
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_google_genai import GoogleGenerativeAI
hf_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
model_name_or_path = "ahmadmac/Trained-T5-large"
# pipe = pipeline(
#     'text2text-generation',
#     model=model_name_or_path,
#     max_length=512,
#     do_sample=True,
#     temperature=1.0
# )
# llm = HuggingFacePipeline(pipeline=pipe)
# prompt_template = """ you are a highly knowledgeable AI assistant. Engage in a conversation with the user. Your main goal is to provide clear and informative answers to the user's questions.
# User: {question}
# Assistant:"""
# prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
# chain = LLMChain(llm=llm, prompt=prompt)
with open("brookline_data.txt", "r") as f:
    data = f.read()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splitted_data = text_splitter.split_text(data)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
metadatas = [{"source": f"source_{i}"} for i in range(len(splitted_data))]
documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(splitted_data, metadatas)]
qdrant = Qdrant.from_documents(
    documents,
    embeddings,
    location=":memory:",
    collection_name="my_documents",
)
retriever = qdrant.as_retriever()
# qna = RetrievalQA.from_chain_type(
#     llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.9, "max_length": 512}, 
#                        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]),
#     chain_type="stuff",
#     retriever=retriever
# )
qna = RetrievalQA.from_chain_type(
    llm=GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["google_api_key"]),
    chain_type="stuff",
    retriever=retriever
)
def chatbot(question, chat_history):
    #response = chain.run(question)
    retrieval_result = qna(question)
    retrieval_answer = retrieval_result['result']
    combined_response = f"Based on the information available:\n{retrieval_answer}"
    return combined_response
demo = gr.ChatInterface(
    fn=chatbot,
    title="Chatbot",
    description="Helpful AI Assistant!"
)
demo.launch()