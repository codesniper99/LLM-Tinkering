from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb
from huggingface_hub import login
from dotenv import load_dotenv
import os
import argparse
import torch

from openai import OpenAI
from pprint import pprint

openai_client = OpenAI(api_key="not-needed", base_url="http://localhost:1234/v1" )

chroma_client = chromadb.PersistentClient(path="./chroma_db/")


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

    embedding_response = openai_client.embeddings.create(
        model="text-embedding-nomic-embed-text-v1.5",
        input=[doc.page_content for doc in docs],
        encoding_format="float"
    )
    print(f"Embedding type: ", type(embedding_response))
    
    pprint(len(embedding_response.data), depth=2, compact=True)
    embeddings = [e.embedding for e in embedding_response.data]
    pprint(len(embeddings[0]))
    return docs, embeddings

def persist_in_collection(docs: List[Document], embeddings):
    print(f"Type of documents: ", type(docs))
    
    collection = chroma_client.get_or_create_collection(
        name="neurips-papers",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [doc.page_content for doc in docs]
    ids = [f"doc-{i}" for i in range(len(docs))]

    assert len(embeddings) == len(docs), "Mismatch between docs and embeddings"
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[doc.metadata for doc in docs]
    )

    return collection

# def setup_openai_llm_chain(vector_store: Chroma, openai_api_key):
#     llm = ChatOpenAI(model="gpt-4o",
#                      temperature=0,
#                      openai_api_key=openai_api_key)
#     prompt = hub.pull("langchain-ai/retrieval-qa-chat")

#     combine_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
#     rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_doc_chain)

#     return rag_chain

# def setup_mistral_llm_chain(vector_store: Chroma, openai_api_key):
#     model_id = "tiiuae/falcon-rw-1b"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id
#     )

#     pipe = pipeline(
#         task="text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         model_kwargs={
#             "max_new_tokens": 300,
#             "temperature": 0.7,
#             "do_sample": True
#         }
#     )


#     llm = HuggingFacePipeline(pipeline=pipe)
    
#     prompt = ChatPromptTemplate.from_template(
#         "You're a concise AI assistant. Given the following context, answer the user's question.\n\n"
#         "Context:\n{context}\n\n"
#         "Question:\n{input}\n"
#     )
#     # prompt = hub.pull("langchain-ai/retrieval-qa-chat")

#     combine_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
#     rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_doc_chain)

#     return rag_chain

# def setup_lm_studio_chain(vector_store: Chroma, openai_api_key):
#     pass

# def retrieve_context():
#     pass

if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description="Choose LLM Backend")
    parser.add_argument("--llm", type=str, choices=["openai", "mistral"], default="mistral", help="Which LLM to use: 'openai' or 'mistral'")
    args = parser.parse_args()
    # print(torch.cuda.is_available())
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0)) 
    
    load_dotenv()
    # openai_api_key = os.getenv("OPENAI_API_KEY")

    pages = load_paper_pdf()
    docs, embeddings = chunk_and_embed_document(pages)
    collection = persist_in_collection(docs=docs, embeddings=embeddings)
    # if args.llm == "openai":
    #     rag_chain = setup_openai_llm_chain(vector_store=vector_store, openai_api_key=openai_api_key)
    # else:
    #     rag_chain = setup_mistral_llm_chain(vector_store=vector_store, openai_api_key=openai_api_key)


    
    
    # # --- Step 6: Ask interactively ---
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-nomic-embed-text-v1.5",
            input=query,
            encoding_format="float"
        )
        query_embedding = embedding_response.data[0].embedding
        
        rag_result = collection.query(
            n_results=5, 
            query_embeddings=query_embedding
            )
        print(type(rag_result['documents']))

        # chunks = [doc for doc in rag_result['documents'][0]]
        chunks = [doc.page_content for doc in docs]
        context = "\n\n".join(chunks)
        prompt = f"""You are an expert assistant. Use the following context which is based on a research paper to answer the question which follows it. Context is wrapped in << and >>, question is wrapped by < and >
        The user's questions will be related to this paper provided in the context.

        Context:
        <<{context}>>

        Question: <{query}>
        Answer:"""
        print(f"Prompt is {prompt}")

        response = openai_client.chat.completions.create(
            model="google/gemma-3-1b",
            messages=[
                {"role": "system", "content": "You are helpful AI Assistant!"},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        print(response)
        print("\n\n Answer: \n")
        ai_response = response.choices[0].message.content
        print(response.choices[0].message.content)
        print("Response received.")

        query_embedding_2 = openai_client.embeddings.create(
            model="text-embedding-nomic-embed-text-v1.5",
            input=ai_response,
            encoding_format="float"
        )
        rag_result_2 = collection.query(n_results=1, query_embeddings=query_embedding_2.data[0].embedding)
        chunks_2 = [doc for doc in rag_result_2['documents'][0]]
        answer = "\n\n".join(chunks_2)
        print(f"Grounding Reference: {answer}")
