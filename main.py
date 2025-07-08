from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
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

def main():
    documents = load_document()
    splitted_docs = chunk_split(documents)
    vector_store = embed_store(splitted_docs)
    




if __name__ == "__main__":
    main()
