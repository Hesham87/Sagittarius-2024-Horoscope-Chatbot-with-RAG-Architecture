import os
from pathlib import Path

import dotenv


from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from llama_cpp import Llama
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import CrossEncoder


class Sagittarius:
    base_dir = Path(__file__).resolve().parent.parent

    data_path = base_dir / "data" / "raw" / "horoscope.txt"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    print(f"Loading data from: {data_path}")

    loader = TextLoader(str(data_path))
    documents = loader.load()

    dotenv.load_dotenv()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=4, separators=["\n\n", "\n", " "]
    )
    docs = text_splitter.split_documents(documents)
    print("loaded docs")
    embeddings = HuggingFaceEmbeddings()
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "langchain-demo"

    if index_name not in pc.list_indexes():
        pc.create_index(name=index_name, metric="cosine", dimension=768, spec=spec)
        docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    else:
        docsearch = pc.from_existing_index(index_name, embeddings)

    def generate_response(data):
        seen = set()
        unique_docs = []
        for doc in data["context"]:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        context_texts: list[str] = [doc.page_content for doc in unique_docs]

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        new_contxt = model.rank(data["question"], context_texts, return_documents=True, top_k=3)

        filtered = [d for d in new_contxt if d["score"] >= -2]

        texts = [d["text"] for d in filtered]

        new_contxt = "\n\n".join(texts)
        print(new_contxt)
        model = Llama(
            model_path="/teamspace/studios/this_studio/Sagittarius-2024-Horoscope-Chatbot-with-RAG-Architecture/models/llama-2-7b-chat.Q4_K_M.gguf",
            n_ctx=4000,
            n_batch=512,
            verbose=False,
        )
        template = """
        you are an astroleger that only knows what is in the context, These Human will ask you questions about the stars. 
        Use following piece of context to answer the question. 
        If you don't know the answer or the answer is not in the context, just say you don't know. 
        Keep the answer within 2 sentences and concise.

        Context: {context}
        Question: {question}
        Answer: 
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        formatted_prompt = prompt.format(context=new_contxt, question=data["question"])

        response = model(formatted_prompt, temperature=0.8, top_k=50, max_tokens=200, stop=["\n"])
        return response["choices"][0]["text"]

    runnable_generate = RunnableLambda(generate_response)

    # Create the chain
    rag_chain = RunnableSequence(
        {
            "context": docsearch.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 10,
                    "fetch_k": 50,
                    "lambda_mult": 0.7,  # Balance: 0=diversity, 1=relevance
                },
            ),
            "question": RunnablePassthrough(),
        },
        runnable_generate,
    )
