# -- coding: utf-8 --
import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
import faiss
import time  # <-- IMPORTED FOR RATE LIMITING
from typing import List, Tuple, Dict

# LangChain & AI Model imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- BACKEND LOGIC ---
# These functions handle the core processing tasks.

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        # To read the file from Streamlit's uploader, we use its byte stream
        file_bytes = uploaded_file.getvalue()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

@st.cache_data
def split_by_study_notes(text: str) -> List[Tuple[str, str]]:
    """Splits the text into chapters based on 'Study Note' headings."""
    # Regex to match "Study Note – 1", "Study Note – 2", etc., with variations
    pattern = r"(Study Note\s*[-–—]\s*\d+[:\s][^\n]*)"
    parts = re.split(pattern, text)
    
    # Filter out empty strings that re.split might produce
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 2:
        return []

    chapters = []
    # Start from the first captured title, assuming the first part might be a preface
    i = 0
    if not parts[0].lower().startswith("study note"):
        i = 1
    
    while i < len(parts) - 1:
        title = parts[i]
        content = parts[i+1]
        chapters.append((title, content))
        i += 2
        
    return chapters

# --- Summarization Functions ---

def get_summarization_chain(llm):
    """Creates a LangChain chain for summarizing a chunk of text."""
    summarize_prompt = PromptTemplate.from_template(
        """You are an expert economics assistant. Your task is to provide a concise summary of the following text from a chapter.
Focus on the key definitions, principles, and economic concepts. The summary should be clear and easy to understand for a student.

Chapter Title: {title}
Content to Summarize:
---
{content}
---

Concise Summary:"""
    )
    return summarize_prompt | llm | StrOutputParser()

def get_combine_chain(llm):
    """Creates a LangChain chain to combine multiple summaries into one."""
    combine_prompt = PromptTemplate.from_template(
        """You are an expert editor. You are given several summaries from different parts of a single chapter.
Your task is to synthesize these partial summaries into a single, cohesive, and detailed final summary of the entire chapter.
Ensure the final summary flows logically and covers all key topics. Use markdown for structure (e.g., headings, bullet points).

Chapter Title: {title}
Partial Summaries:
---
{content}
---

Detailed Final Summary:"""
    )
    return combine_prompt | llm | StrOutputParser()

# MODIFIED FUNCTION TO HANDLE RATE LIMITING
@st.cache_data(show_spinner=False) # Spinner is handled manually now with a progress bar
def summarize_chapter(_api_key: str, title: str, content: str) -> str:
    """
    Summarize a long chapter using Gemma (Groq) with a map-reduce approach.
    This version processes chunks sequentially with a delay to respect API rate limits.
    """
    # This key is passed to ensure st.cache_data re-runs if the key changes,
    # but it's used directly in the LLM initialization.
    llm = ChatGroq(api_key="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", model_name="gemma2-9b-it", temperature=0.2)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
    chunks = text_splitter.split_text(content)
    
    st.info(f"Chapter '{title}' has been split into {len(chunks)} chunks. Summarizing each one...")
    
    # --- START OF MODIFIED SECTION ---
    # Process chunks sequentially to avoid hitting the 15k tokens/minute rate limit.
    
    # --- Summarization Functions ---

def get_summarization_chain(llm):
    """Creates a LangChain chain for summarizing a chunk of text."""
    summarize_prompt = PromptTemplate.from_template(
        """You are an expert economics assistant. Your task is to provide a concise summary of the following text from a chapter.
Focus on the key definitions, principles, and economic concepts. The summary should be clear and easy to understand for a student.

Chapter Title: {title}
Content to Summarize:
---
{content}
---

Concise Summary:"""
    )
    return summarize_prompt | llm | StrOutputParser()

def get_combine_chain(llm):
    """Creates a LangChain chain to combine multiple summaries into one."""
    combine_prompt = PromptTemplate.from_template(
        """You are an expert editor. You are given several summaries from different parts of a single chapter.
Your task is to synthesize these partial summaries into a single, cohesive, and detailed final summary of the entire chapter.
Ensure the final summary flows logically and covers all key topics. Use markdown for structure (e.g., headings, bullet points).

Chapter Title: {title}
Partial Summaries:
---
{content}
---

Detailed Final Summary:"""
    )
    return combine_prompt | llm | StrOutputParser()

# --- NEW, HIERARCHICAL VERSION OF summarize_chapter ---
@st.cache_data(show_spinner=False) # Spinner is handled manually now with a progress bar
def summarize_chapter(_api_key: str, title: str, content: str) -> str:
    """
    Summarize a long chapter using a hierarchical map-reduce approach to avoid context length errors.
    This version processes chunks sequentially with a delay to respect API rate limits.
    """
    llm = ChatGroq(api_key="your_groq_api", model_name="llama3-70b-8192", temperature=0.2)
    
    # === 1. MAP STEP: Summarize initial chunks ===
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
    chunks = text_splitter.split_text(content)
    
    st.info(f"Chapter '{title}' has been split into {len(chunks)} chunks. Summarizing each one...")
    
    summarize_chain = get_summarization_chain(llm)
    partial_summaries = []
    
    # Add a progress bar for better user experience
    progress_bar = st.progress(0, text="Summarizing chunks...")

    for i, chunk in enumerate(chunks):
        try:
            summary = summarize_chain.invoke({"title": title, "content": chunk})
            partial_summaries.append(summary)
            progress_text = f"Summarizing chunk {i + 1}/{len(chunks)}..."
            progress_bar.progress((i + 1) / len(chunks), text=progress_text)
            time.sleep(6) # CRITICAL: Respect rate limits
        except Exception as e:
            st.error(f"Error summarizing chunk {i+1}: {e}")
            partial_summaries.append(f"[Error summarizing chunk {i+1}]")
            continue
            
    progress_bar.empty()

    if not partial_summaries:
        return "Could not generate any summaries for the chapter chunks."

    if len(partial_summaries) == 1:
        st.success("Summary generated!")
        return partial_summaries[0]
        
    # === 2. REDUCE STEP: Combine summaries hierarchically ===
    combine_chain = get_combine_chain(llm)
    current_summaries = partial_summaries
    combine_batch_size = 8 # Number of summaries to combine at once. Keep this low.

    # Loop until we have only one summary left
    while len(current_summaries) > 1:
        st.info(f"Combining {len(current_summaries)} summaries in batches of {combine_batch_size}...")
        
        next_level_summaries = []
        progress_bar = st.progress(0, text="Combining summaries...")
        
        # Group current summaries into batches
        for i in range(0, len(current_summaries), combine_batch_size):
            batch = current_summaries[i:i + combine_batch_size]
            if not batch:
                continue

            combined_content_for_batch = "\n\n---\n\n".join(batch)
            
            try:
                # Combine this small batch
                new_summary = combine_chain.invoke({"title": title, "content": combined_content_for_batch})
                next_level_summaries.append(new_summary)
                
                # Update progress
                progress_value = (i + len(batch)) / len(current_summaries)
                progress_text = f"Combined batch {i//combine_batch_size + 1}/{ -(-len(current_summaries) // combine_batch_size) }"
                progress_bar.progress(progress_value, text=progress_text)

                time.sleep(6) # CRITICAL: Respect rate limits for each combine call too
            except Exception as e:
                st.error(f"Error combining summary batch: {e}")
                next_level_summaries.append("[Error during combination step]")
                continue

        progress_bar.empty()
        current_summaries = next_level_summaries # The new, smaller list of summaries becomes the input for the next loop iteration

    st.success("Final summary generated!")
    return current_summaries[0]
# --- RAG (Chat) Functions ---

@st.cache_resource
def get_embedding_model():
    """Loads and caches the sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Building retrieval indexes for all chapters...")
def build_rag_indexes(_chapters: List[Tuple[str, str]]) -> Tuple[Dict, Dict]:
    """Builds FAISS indexes for each chapter for RAG."""
    chapter_chunks = {}
    chapter_indexes = {}
    embedding_model = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for title, content in _chapters:
        chunks = text_splitter.split_text(content)
        if not chunks:
            continue
        
        embeddings = embedding_model.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))

        chapter_chunks[title] = chunks
        chapter_indexes[title] = index
    
    return chapter_chunks, chapter_indexes

def get_rag_answer(llm, question: str, selected_chapter_title: str, chapter_chunks: Dict, chapter_indexes: Dict) -> str:
    """Retrieves context and generates an answer using the RAG pipeline."""
    if selected_chapter_title not in chapter_indexes:
        return "This chapter has no content to search or could not be processed."
        
    index = chapter_indexes[selected_chapter_title]
    chunks = chapter_chunks[selected_chapter_title]
    embedding_model = get_embedding_model()

    question_embedding = embedding_model.encode([question])
    
    # Search for top 3 most relevant chunks
    _, I = index.search(np.array(question_embedding).astype("float32"), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = PromptTemplate.from_template(
        """You are a helpful study assistant. Based ONLY on the provided context from the textbook, answer the user's question.
If the context does not contain the answer, state that the information is not available in the provided text. Do not make up information.

Context from the Chapter:
---
{context}
---

Question: {question}

Answer:"""
    )

    rag_chain = prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({"context": context, "question": question})
    return answer

# --- STREAMLIT UI ---

def main():
    st.set_page_config(page_title="RAG Study Assistant", layout="wide")

    # --- Session State Initialization ---
    if "chapters" not in st.session_state:
        st.session_state.chapters = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False
    if "chapter_chunks" not in st.session_state:
        st.session_state.chapter_chunks = {}
    if "chapter_indexes" not in st.session_state:
        st.session_state.chapter_indexes = {}

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Setup")
        st.markdown("An interactive RAG application for studying/interacting with specific subject domain textbooks.")
        
        # Hardcoding the key for simplicity based on original code, but still using input field
        # For security, using text_input with type="password" is best practice.
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password", value="your_groq_api")

        uploaded_file = st.file_uploader(
            "Upload your Textbook PDF", type="pdf",
            help="The PDF will be processed to extract chapters based on 'Study Note' headings."
        )

        if uploaded_file:
            # Process the file only if chapters aren't already in session state
            if not st.session_state.chapters:
                with st.spinner("Processing PDF..."):
                    full_text = extract_text_from_pdf(uploaded_file)
                    if full_text:
                        st.session_state.chapters = split_by_study_notes(full_text)
                        if st.session_state.chapters:
                            st.success(f"✅ Successfully extracted {len(st.session_state.chapters)} chapters!")
                        else:
                            st.warning("⚠️ Could not find chapters structured with 'Study Note – X'. Please check the PDF format.")

        st.markdown("---")
        st.info("Done.")

    # --- Main Page Content ---
    st.title("📚 Domain-Focused RAG Study Assistant")
    
    if not groq_api_key:
        st.info("👈 Please enter your Groq API Key in the sidebar to begin.")
        st.stop()
        
    if not st.session_state.chapters:
        st.info("👈 Please upload a PDF in the sidebar to get started.")
        st.stop()

    # Initialize LLM once the key is available
    try:
        llm = ChatGroq(api_key="your_groq_api", model="gemma2-9b-it", temperature=0.3)
    except Exception as e:
        st.error(f"❌ Failed to initialize the language model. Please check your API key. Error: {e}")
        st.stop()
        
    chapter_titles = [title for title, _ in st.session_state.chapters]
    
    tab1, tab2 = st.tabs(["✨ Chapter Summarizer", "💬 Chat with a Chapter (RAG)"])
    
    # --- Summarizer Tab ---
    with tab1:
        st.header("Generate Chapter Summaries")
        st.markdown("Select a chapter to generate a detailed summary. This uses a map-reduce process for long texts and caches the result.")
        
        selected_title_summ = st.selectbox(
            "Choose a chapter to summarize:",
            options=chapter_titles,
            index=0,
            key="summarizer_select"
        )

        if st.button("Generate Summary"):
            if selected_title_summ:
                # Check cache first
                if selected_title_summ in st.session_state.summaries:
                    st.info("Displaying cached summary.")
                    st.markdown(f"### Summary for {selected_title_summ}")
                    st.markdown(st.session_state.summaries[selected_title_summ], unsafe_allow_html=True)
                else:
                    selected_content = [content for title, content in st.session_state.chapters if title == selected_title_summ][0]
                    # Pass the key to satisfy cache requirements
                    summary = summarize_chapter(groq_api_key, selected_title_summ, selected_content)
                    st.session_state.summaries[selected_title_summ] = summary
                    st.markdown(f"### Summary for {selected_title_summ}")
                    st.markdown(summary, unsafe_allow_html=True)

    # --- RAG Chat Tab ---
    with tab2:
        st.header("Chat with a Specific Chapter")
        st.markdown("Ask questions about a specific chapter. The system will retrieve relevant information to form an answer.")
        
        if not st.session_state.rag_ready:
            st.write("The chat system needs to prepare the document for efficient searching.")
            if st.button("🚀 Prepare Chat System"):
                chunks, indexes = build_rag_indexes(st.session_state.chapters)
                st.session_state.chapter_chunks = chunks
                st.session_state.chapter_indexes = indexes
                st.session_state.rag_ready = True
                st.rerun() # Rerun to show the chat interface

        if st.session_state.rag_ready:
            st.success("✅ Chat system is ready!")
            
            rag_selected_title = st.selectbox(
                "Choose a chapter to ask questions about:",
                options=chapter_titles,
                index=0,
                key="rag_select"
            )

            user_question = st.text_input("What is your question?")
            
            if user_question:
                with st.spinner("Finding answer..."):
                    answer = get_rag_answer(
                        llm,
                        user_question,
                        rag_selected_title,
                        st.session_state.chapter_chunks,
                        st.session_state.chapter_indexes
                    )
                    st.markdown("### Answer")
                    st.markdown(answer)

if __name__ == "__main__":
    main()
