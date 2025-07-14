import queue
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel, EncDecCTCModelBPE 
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import chromadb
from huggingface_hub import login
from dotenv import load_dotenv
import os
import numpy as np
import sounddevice as sd
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
def get_question_by_text():
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        return "exit"
    return query


def generate_rag_response(query, collection, docs):
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
    # print(type(rag_result['documents']))

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
        temperature=0,
        stream=True
    )
    print("\n\nAnswer (streaming):\n")
    for chunk in response:
        # print(chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            # print(token, end="", flush=True)
            yield token

def get_query_by_audio():
    
    print("Inside")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b", map_location=device)
    

    print(f"ASR Model {asr_model}")
    query = "What is this paper about"
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 1024
    CHANNELS = 1
    BUFFER_DURATION = 1.0  # seconds of audio per inference
    OVERLAP = 0.5          # seconds overlap between chunks
    print(dir(asr_model))
    audio_q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print("Stream status:", status)
        audio_q.put(indata.copy())

    buffer = np.zeros(int(SAMPLE_RATE * BUFFER_DURATION), dtype=np.float32)
    overlap_samples = int(OVERLAP * SAMPLE_RATE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype='float32', callback=audio_callback):
        print("🎤 Listening... Ctrl+C to stop")
        while True:
            try:
                chunk = audio_q.get()
                chunk = chunk.flatten()

                # Roll buffer and append new data
                buffer = np.roll(buffer, -len(chunk))
                buffer[-len(chunk):] = chunk

                # Run transcription every BUFFER_DURATION seconds
                audio_input = buffer.copy()
                text = asr_model.transcribe([audio_input])[0] # type:  ignore
                print("🗣️", text)

            except KeyboardInterrupt:
                print("🛑 Stopped.")
                break

    return query
def argument_parsing():
    parser  = argparse.ArgumentParser(description="Choose LLM Backend")
    parser.add_argument("--llm", type=str, choices=["openai", "mistral"], default="mistral", help="Which LLM to use: 'openai' or 'mistral'")
    args = parser.parse_args()
    return args

if __name__ == "__main__":  
    print(sd.query_devices())
    args = argument_parsing()
    load_dotenv()
    # openai_api_key = os.getenv("OPENAI_API_KEY")

    pages = load_paper_pdf()
    docs, embeddings = chunk_and_embed_document(pages)
    collection = persist_in_collection(docs=docs, embeddings=embeddings)

    # query = get_query_by_audio()
    # # --- Step 6: Ask interactively ---
    while True:
        query = get_question_by_text()
        

        
        if query == "exit":
            break
        
        # generate_rag_response(query=query)
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
        # print(type(rag_result['documents']))

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
            temperature=0,
            stream=True
        )
        streamed_response = ""
        print("\n\nAnswer (streaming):\n")
        for chunk in response:
            # print(chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                # yield token
                streamed_response += token
                
        # print("\n\nResponse received.")
        # print(response)
        # print("\n\n Answer: \n")
        # ai_response = response.choices[0].message.content
        # print(response.choices[0].message.content)
        # print("Response received.")

        # query_embedding_2 = openai_client.embeddings.create(
        #     model="text-embedding-nomic-embed-text-v1.5",
        #     input=ai_response,
        #     encoding_format="float"
        # )
        # rag_result_2 = collection.query(n_results=1, query_embeddings=query_embedding_2.data[0].embedding)
        # chunks_2 = [doc for doc in rag_result_2['documents'][0]]
        # answer = "\n\n".join(chunks_2)
        # print(f"Grounding Reference: {answer}")
