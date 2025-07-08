from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import os
import shutil
import sys

file_path = '''data'''
db_dir = '''./chroma_langchain_db'''
recreate = True

def load_document():
    try:
        pages = []
        for filename in os.listdir(file_path):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path=os.path.join(file_path, filename))
                for page in loader.lazy_load():
                    pages.append(page)
        # print(f"{pages[0].metadata}\n")
        # print(pages[0].page_content)   
        return pages
    except Exception as e:
        print("Exception occured while loading document")

def chunk_split(documents):
    try:
        texts = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        id_count = 0
        for i in documents:
            text = text_splitter.split_text(i.page_content)
            for j in text:
                doc = Document(
                    page_content=j,
                    metadata = i.metadata,
                    id=id_count
                )
                id_count = id_count+1
                texts.append(doc)
        # for docs in texts:
        #     print("ID: " + docs.id)
        #     print("Metadata: ", end=": ")
        #     print(docs.metadata)
        #     print("Content: " + docs.page_content)
        #     print("\n\n\n\n")
        return texts
    except Exception as e:
        print("Exception occured while splitting")


def embed_store(documents, collection_name="test_collection", model_name = "all-mpnet-base-v2"):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{model_name}", 
            model_kwargs = {'device': 'cpu'})
        if not os.path.exists(db_dir) or recreate:
            print("Creating new database.....\n\n")
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory="./chroma_langchain_db"
            )

            vector_store.add_documents(documents=documents)
        else:
            print("Old database found !!! \n\n")
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory="./chroma_langchain_db"
            )
        
        return vector_store
    except Exception as e:
        print("Exception occured while embedding and storing. " + str(e))
        sys.exit(1)

def retrieve_docs_db(vector_store, question):
    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k':3})
        fetched_docs = retriever.invoke(question)
        # for i, data in enumerate( fetched_docs):
        #     print(f"Document {i}:")
        #     print(data.page_content)
        #     print("\n\n")
        return fetched_docs
    except Exception as e:
        print("Exception during retrieval of related docs")
        sys.exit(1)

def response_from_llm(relevant_docs, question):
    try:
        llm = ChatOllama(model="gemma3:1b")
        system_prompt = "You are a question-answer bot. You have one simple job, you need to answer to questions consicely based on the following context. Context: {context}.\n\n\n\n Special Instructions: Remember your answer should be brief and it should be to the point to the question. If you do not find the answer in the context provided tell that you don't know the answer as it is not present in the context. "

        context_str = "".join(d.page_content for d in relevant_docs)
        contextual_prompt = system_prompt.format(context=context_str)

        response = llm.invoke([SystemMessage(content=contextual_prompt), 
                               HumanMessage(content=question)])
        return response
    except Exception as e:
        print("Exception in getting response from llm")
        sys.exit(1)

def main():
    if not os.path.exists(db_dir) or recreate:
        documents = load_document()
        splitted_docs = chunk_split(documents)
    else:
        splitted_docs=None

        
    vector_store = embed_store(splitted_docs)

    questions = ["What was Arjun known for in the village?", 
                 "What did Arjun find in the abandoned hut?",
                 "What was the name of the book that Arjun found in the abandoned hut?",
                 "What did Arjun do after learning from the book?", 
                 "How did the village change due to Arjun's efforts?",
                 "What lesson did Arjun teach the villagers?",
                 "Who discovered the book?"
                 ]
    for index, question in enumerate(questions):
        relevant_docs = retrieve_docs_db(vector_store, question)
        answer = response_from_llm(relevant_docs, question)
        print(f"Question {index+1}: " + question)
        print("Answer: " + answer.content)
        print("\n\n")

    # while True:
    #     question = input("Enter the question: ")
    #     relevant_docs = retrieve_docs_db(vector_store, question)
    #     answer = response_from_llm(relevant_docs, question)
    #     print("Answer: " + answer.content)
    #     print("\n\n")

    




if __name__ == "__main__":
    main()
