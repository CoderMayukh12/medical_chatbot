from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#loading raw pdf

DATA_PATH="data/"
def load_pdf(data):
    loader= DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents= loader.load()
    return documents
documents= load_pdf(data= DATA_PATH)
#print("length:", len(documents))

#create chunks 

def create_chunks(extracted_data):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50) #overlap-> each new chunk starts by repeating a small part from the end of the previous one
                                                                                    #so every detail in covered and the context remain intact
    text_chunks= text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks= create_chunks(extracted_data=documents)
#print("length of text chunks:",len(text_chunks))

#create vector embedding of the text chunks

def get_embedding_model():
    embedding_model= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model= get_embedding_model()

#store embeddings in FAISS
DB_FAISS_PATH= "vectorstore/db_faiss"
db= FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)

