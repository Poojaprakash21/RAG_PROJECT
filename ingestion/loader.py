from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_documents(pdf_dir):
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    return documents
