from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = '''data\\Arjun, the nature lover.pdf'''

def load_document():
    loader = PyPDFLoader(file_path=file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    # print(f"{pages[0].metadata}\n")
    # print(pages[0].page_content)   
    return pages

def chunk_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_text(documents[0].page_content)
    return texts

def main():
    documents = load_document()
    splitted_docs = chunk_split(documents)
    # for i in splitted_docs:
        # print(i + "*******\n\n\n\n")


if __name__ == "__main__":
    main()
