from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
from dotenv import load_dotenv
import os
import argparse
import torch


def load_paper_pdf():
    print(f"Loading Papers PDFs")
    pdf_path = "../data/paper1.pdf"
    pdf_name = os.path.basename(pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for doc in pages:
        doc.metadata["source_pdf"] = pdf_name

    print(f"Number of pages in PDF ", len(pages))
    return pages

def chunk_and_embed_document(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    print(f"Docs length: ", len(docs))
    embeddings = HuggingFaceEmbeddings()
    print(f"Embedding type: ", type(embeddings))
    return docs, embeddings

def persist_in_vector_store(docs, embeddings):
    print(f"Type of documents: ", type(docs))
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="neurips-papers")

    # check for existing pdfs
    existing = vector_store.get(include=["metadatas"])
    
    pdf_path = "../data/paper1.pdf"
    pdf_name = os.path.basename(pdf_path)
    already_inserted = (metadata.get("source_pdf") == pdf_name for metadata in existing["metadatas"])
    if already_inserted:
        print(f"{pdf_name} already exists in  vector store, skipping it")
    else:
        print(f"Adding {pdf_name} to the vector store ")
        vector_store.add_documents(documents=docs)
    return vector_store

def setup_openai_llm_chain(vector_store: Chroma, openai_api_key):
    llm = ChatOpenAI(model="gpt-4o",
                     temperature=0,
                     openai_api_key=openai_api_key)
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_doc_chain)

    return rag_chain

def setup_mistral_llm_chain(vector_store: Chroma, openai_api_key):
    model_id = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id
    )

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={
            "max_new_tokens": 300,
            "temperature": 0.7,
            "do_sample": True
        }
    )


    llm = HuggingFacePipeline(pipeline=pipe)
    
    prompt = ChatPromptTemplate.from_template(
        "You're a concise AI assistant. Given the following context, answer the user's question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{input}\n"
    )
    # prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_doc_chain)

    return rag_chain

def setup_lm_studio_chain(vector_store: Chroma, openai_api_key):
    pass
def retrieve_context():
    pass

if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description="Choose LLM Backend")
    parser.add_argument("--llm", type=str, choices=["openai", "mistral"], default="mistral", help="Which LLM to use: 'openai' or 'mistral'")
    args = parser.parse_args()

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    pages = load_paper_pdf()
    docs, embeddings = chunk_and_embed_document(pages)
    vector_store = persist_in_vector_store(docs=docs, embeddings=embeddings)
    if args.llm == "openai":
        rag_chain = setup_openai_llm_chain(vector_store=vector_store, openai_api_key=openai_api_key)
    else:
        rag_chain = setup_mistral_llm_chain(vector_store=vector_store, openai_api_key=openai_api_key)


    
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0)) 
    # --- Step 6: Ask interactively ---
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break

        result = rag_chain.invoke({"input": query, "context": docs})
        print("\nAnswer:", result["answer"])
        print("Response received.")