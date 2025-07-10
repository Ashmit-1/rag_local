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
from uuid import uuid4

file_path = '''data'''
db_dir = '''./chroma_langchain_db'''
recreate = False

def load_document(file_path):
    try:
        pages = []
        if os.path.isdir(file_path):
            for filename in os.listdir(file_path):
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path=os.path.join(file_path, filename))
                    for page in loader.lazy_load():
                        pages.append(page)
        elif os.path.isfile(file_path):
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path=file_path)
                for page in loader.lazy_load():
                    pages.append(page)
        # print(f"{pages[0].metadata}\n")
        # print(pages[0].page_content)   
        return pages
    except Exception as e:
        print("Exception occured while loading document. " + str(e))
        sys.exit(1)

def chunk_split(documents):
    try:
        texts = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        for i in documents:
            text = text_splitter.split_text(i.page_content)
            for j in text:
                doc = Document(
                    page_content=j,
                    metadata = i.metadata,
                    id=str(uuid4())
                )
                texts.append(doc)
        # for docs in texts:
        #     print("ID: " + docs.id)
        #     print("Metadata: ", end=": ")
        #     print(docs.metadata["source"])
        #     print("Content: " + docs.page_content)
        #     print("\n\n\n\n")
        return texts
    except Exception as e:
        print("Exception occured while splitting. " + str(e))
        sys.exit(1)


def embed_store(collection_name = "test_collection", model_name = "bge-small-en-v1.5"):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=f"BAAI/{model_name}", 
            model_kwargs = {'device': 'cpu'},
            encode_kwargs={"normalize_embeddings": True})
        if not os.path.exists(db_dir) or recreate:
            print("Creating new database.....\n")
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory="./chroma_langchain_db"
            )

            # uuids = [str(uuid4()) for _ in range(len(documents))]
            # vector_store.add_documents(documents=documents, ids=uuids)
        else:
            print("Old database found !!! \n")
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory="./chroma_langchain_db"
            )
        
        return vector_store
    except Exception as e:
        print("Exception occured while embedding and storing. " + str(e))
        sys.exit(1)

def auto_sync_db(vector_store):
    try:

        print("Syncing....")

        ### this is for getting the files which are there in the directory
        data_dir_set = set()
        for filename in os.listdir(file_path):
            if filename.endswith(".pdf"):
                data_dir_set.add(os.path.join(file_path, filename))
        
        ### this is for getting the files already present in the database
        database_set = set()
        docs = vector_store.get()
        source_id_dict = {}
        for index in range(len(docs["ids"])):
            database_set.add(docs["metadatas"][index]["source"])
            source_id_dict.setdefault(docs["metadatas"][index]["source"], []).append(docs["ids"][index])
            # print("Id: " + docs["ids"][index])
            # print("Source: " + docs["metadatas"][index]["source"])
            # print("Author: " + docs["metadatas"][index]["author"])
            # print("Content: " + docs["documents"][index])
            # print("\n\n" + "*" * 100)
        # for i in source_id_dict:
        #     print(f"{i} : {source_id_dict[i]}\n\n")


        ### adding newly added files to the database
        to_add = data_dir_set - database_set
        try:
            for new_file in to_add:
                print(f"Adding {new_file} in the database....")
                documents = load_document(new_file)
                splitted_docs = chunk_split(documents)
                uuids = [str(uuid4()) for _ in range(len(splitted_docs))]
                vector_store.add_documents(documents=splitted_docs, ids=uuids)
        except Exception as e:
            print("Exception occured from the sync function while adding new files to the database. " + str(e))
            sys.exit(1)

        
        ### deleting data in order to sync with the data directory
        to_delete = database_set - data_dir_set
        try:
            for deleted_file in to_delete:
                print(f"Deleting {deleted_file} from the database...")
                vector_store.delete(ids=source_id_dict[deleted_file])
        except Exception as e:
            print("Exception occured from the sync function while deleting data from database. " + str(e))
        
        print("Successfully synced !!")

    except Exception as e:
        print("Exception occured while smart sync. " + str(e))
        sys.exit(1)

def retrieve_docs_db(vector_store, question):
    try:
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k':5})
        fetched_docs = retriever.invoke(question)

        # for i, data in enumerate( fetched_docs):
        #     print(f"Document {i}:")
        #     print(data.page_content)
        #     print("\n\n")
        
        return fetched_docs
    except Exception as e:
        print("Exception during retrieval of related docs")
        sys.exit(1)

def response_from_llm(relevant_docs, questions):
    try:
        llm = ChatOllama(model="gemma3:1b")
        system_prompt = "You are a question-answer bot. You have one simple job, you need to answer to questions consicely based on the following context. Context: {context}.\n\n\n\n Special Instructions: Remember your answer should be brief and it should be to the point to the question. If you do not find the answer in the context provided tell that you don't know the answer as it is not present in the context. "

        context_str = "".join(d.page_content for d in relevant_docs)
        contextual_prompt = system_prompt.format(context=context_str)

        response = llm.invoke([SystemMessage(content=contextual_prompt), 
                               HumanMessage(content=questions)])
        return response
    except Exception as e:
        print("Exception in getting response from llm. " + str(e))
        sys.exit(1)

def main():
    vector_store = embed_store()
    auto_sync_db(vector_store)

    questions = ["What was Arjun known for in the village?", 
                 "What did Arjun find in the abandoned hut?",
                 "What was the name of the book that Arjun found in the abandoned hut?",
                 "What did Arjun do after learning from the book?", 
                 "How did the village change due to Arjun's efforts?",
                 "What lesson did Arjun teach the villagers?",
                 "What made Master Elric\'s clocks special?",
                 "Why did Elric take Ravi in?",
                 "What was unique about the hidden clock?",
                 "What did Ravi use to complete the clock?",
                 "What did the clock reveal when it was completed?"                
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
