import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt
@st.cache_resource
def load_local_llm():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", torch_dtype=torch.float16).to("cuda")
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=256, truncation=True, temperature=0.4)
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    return llm


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])


    prompt= st.chat_input("pass your prompt here")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})


        CUSTOM_PROMPT_TEMPLATE = """
        Answer concisely based on the provided context.
        Context: {context}
        Question: {question}
        """
        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_local_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=False,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            #response="Hi, I am braineeBOT!"
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 
