from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

file_path = '''data\\Arjun, the nature lover.pdf'''

def load_document():
    try:
        loader = PyPDFLoader(file_path=file_path)
        pages = []
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
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

def embed_data(data):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cpu'})
        embed_text = embeddings.embed_documents(data)
        # print(len(embed_text))
        return embed_text
    except Exception as e:
        print("Exception occured while embedding \n")

def embed_store(documents, collection_name="test_collection"):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs = {'device': 'cpu'})
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db"
        )

        vector_store.add_documents(documents=documents)
        return vector_store
    except Exception as e:
        print("Exception occured while embedding and storing")

def retrieve_docs_db(vector_store, question):
    try:
        retriever = vector_store.as_retriever(search_kwargs={'k':3})
        fetched_docs = retriever.invoke(question)
        # for i, data in enumerate( fetched_docs):
        #     print(f"Document {i}:")
        #     print(data.page_content)
        #     print("\n\n")
        return fetched_docs
    except Exception as e:
        print("Exception during retrieval of related docs")

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

def main():
    documents = load_document()
    splitted_docs = chunk_split(documents)
    vector_store = embed_store(splitted_docs)
    questions = ["What was Arjun known for in the village?", 
                 "What did Arjun find in the abundunt hut?",
                 "What did Arjun do after learning from the book?", 
                 "How did the village change due to Arjun's efforts?",
                 "What lesson did Arjun teach the villagers?"]
    for question in questions:
        relevant_docs = retrieve_docs_db(vector_store, question)
        answer = response_from_llm(relevant_docs, question)
        print("Question: " + question)
        print("Answer: " + answer.content)
        print("\n\n")
    




if __name__ == "__main__":
    main()
