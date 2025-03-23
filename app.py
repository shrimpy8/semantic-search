import os
import uuid
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Initialize session state to track app state
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = False

st.title("Semantic Search Engine")
st.header("Upload a file to get started.")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")   

# Chroma vector store
chroma_vector_store = Chroma(
    collection_name="semantic_search_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma/db"
)

def clear_vector_store():
    try:
        global chroma_vector_store  
        chroma_vector_store.delete_collection()
        chroma_vector_store = Chroma(
            collection_name="semantic_search_docs",
            embedding_function=embedding_model,
            persist_directory="./chroma/db"
        )
        st.success("Vector store cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")

# UI for clearing vector store
if st.checkbox("Clear existing documents before uploading"):
    clear_vector_store()

# Check current database status
try:
    collection_stats = chroma_vector_store._collection.count()
    if collection_stats > 0:
        st.info(f"Database currently contains {collection_stats} document chunks.")
except Exception as e:
    st.warning(f"Could not check database status: {str(e)}")

# LLM Model
llm_model = ChatOpenAI(model="gpt-4o-mini")

uploaded_file = st.file_uploader("Select a file: ")

if uploaded_file is not None:
    # Add file type validation
    if not uploaded_file.name.lower().endswith('.pdf'):
        st.error("Only PDF files are currently supported.")
    else:
        with st.spinner("Processing file..."):
            try:
                print("File info: ", uploaded_file)

                # Create a safe temporary file with unique name
                file_extension = os.path.splitext(uploaded_file.name)[1]
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")
                
                st.info(f"Processing file: {uploaded_file.name}")

                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # PDF file loader
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
                # Create Text splitter
                chunks = text_splitter.split_documents(docs)
                st.success(f"Document split into {len(chunks)} chunks")
                print("Chunks created: ", len(chunks))

                # For each chunk, print the size
                for i, chunk in enumerate(chunks):
                    print(f"Chunk {i} is of size ", len(chunk.page_content))    
                
                # Display chunk information in an expander
                with st.expander("View document chunk details", expanded=False):
                    for i, chunk in enumerate(chunks):
                        st.write(f"Chunk {i+1} contains {len(chunk.page_content)} characters")
                
                # Index embeddings
                with st.spinner("Creating embeddings and indexing document..."):
                    chroma_ids = chroma_vector_store.add_documents(documents=chunks)
                    st.success(f"Document indexed successfully with {len(chroma_ids)} embeddings")

                # Update session state to indicate file has been processed
                st.session_state.processed_file = True
                print("Chroma IDs: ", chroma_ids)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                print(f"Error details: {e}")
            
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

# Create a retriever                                         
retriever = chroma_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve more chunks
)

# Chat interface
st.subheader("Ask questions about your document")

# Only allow chat if a document has been processed
if prompt := st.chat_input("Ask a question about your document"):
    if not st.session_state.processed_file:
        st.error("Please upload a document before asking questions.")
    else:
        st.write(f"**Question:** {prompt}")
        with st.spinner("Searching for answer..."):
            try:
                docs_retrieved = retriever.invoke(prompt)

                if not docs_retrieved:
                    st.warning("No relevant information found in the document. Try rephrasing your question.")
                else:
                    # Add context display for transparency
                    with st.expander("View context used for answering", expanded=False):
                        for i, doc in enumerate(docs_retrieved):
                            st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")
                    
                    doc_content = "\n\n".join([doc.page_content for doc in docs_retrieved])
                    
                    # Create a Prompt template
                    system_prompt = ''' 
                    You're a helpful assistant. 
                    Please answer the following question {question} only using the following information {document}. 
                    If you can't answer the question, just say you can't answer that question.
                    '''
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt)
                    ])

                    final_prompt = prompt_template.invoke({
                        "question": prompt,
                        "document": doc_content  # Pass only the text content
                    })
                    
                    print("==>Final prompt: ", final_prompt)

                    # Display the result
                    st.subheader("Answer:")
                    result_placeholder = st.empty()
                    
                    # Streaming the completion result
                    full_completion = ""
                    for chunk in llm_model.stream(final_prompt):
                        full_completion += chunk.content
                        result_placeholder.write(full_completion)

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                print(f"Error details: {e}")

# Help section for users
with st.expander("How this app works", expanded=False):
    st.markdown("""
    1. **Upload a PDF document**: The app processes the PDF and splits it into smaller chunks.
    2. **Document indexing**: Each chunk is converted into a vector embedding and stored in a database.
    3. **Ask questions**: When you ask a question, the app finds the most relevant chunks using similarity search.
    4. **Generate answers**: The app uses an AI model to create an answer based only on the retrieved chunks.
    
    **Tips for better results:**
    - Ask specific questions that might be answered in the document
    - If you get "I can't answer that question" responses, try rephrasing your question
    - For complex documents, consider clearing the database before uploading a new file
    """)