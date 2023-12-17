import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
# from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import sentence_transformers
# from sentence_transformers import SentenceTransformer,util
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders.parsers.pdf import PyPDFParser

import pinecone 
from langchain.vectorstores import Pinecone

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import os

load_dotenv(".streamlit/secrets.toml")
key=os.environ["openai_api_key"]
pinecone_key =os.environ["PINECONE_API_KEY"]
pinecone_env =os.environ["PINECONE_ENVIRONMENT"]
pinecone_index =os.environ["PINECONE_INDEX"]
# =os.getenv["openai_api_key"]

model_name = "gpt-4"
llm = OpenAI(api_key=key,model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")

directory="./content/data/*.pdf"
def load_docs(directory):
  loader = get_pdf_text("./content/data/*.pdf")
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = get_pdf_text(documents)
st.write(len(docs))

embeddings = OpenAIEmbeddings(model_name="ada")

query_result = embeddings.embed_query("Hello world")
len(query_result)

pinecone.init(
    api_key=pinecone_key,  # find at app.pinecone.io
    environment=pinecone_env # next to api key in console
)

index_name = pinecone_index

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

def get_similiar_docs(query,k=2,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

query = "How is india's economy"
similar_docs = get_similiar_docs(query)
similar_docs

chain = load_qa_chain(llm, chain_type="stuff")
def get_answer(query):
  similar_docs = get_similiar_docs(query)
  # print(similar_docs)
  answer =  chain.run(input_documents=similar_docs, question=query)
  return  answer

query = "How is india's economy"  
get_answer(query)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    # load_dotenv()
    load_dotenv(".streamlit/secrets.toml")
    st.set_page_config(page_title="Chat with Premium Management PDFs 👇",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ""

    st.header("Ask a question about Topic of Premium Management :books:")
    user_question = st.text_input("Ask a question about Topic of Premium Management :")
    if user_question:
        handle_userinput(user_question)
        query = "How relations between india and us has improved?"
        get_answer(query)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your Premium Management PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                # st.text_input(disabled=False)


if __name__ == '__main__':
    main()
