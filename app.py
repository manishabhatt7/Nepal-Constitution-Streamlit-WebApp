# -----------------------------------------------------------------------------
# CONSTITUTION CHATBOT - A Streamlit Application
# -----------------------------------------------------------------------------

# Import necessary tools and libraries
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # Ignore some less critical warnings

import fitz  # For reading PDF files (PyMuPDF)
from langchain_core.documents import Document # A standard way to represent a piece of text
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as qdrant_exceptions
from openai import APIError, AuthenticationError, APIConnectionError
from langchain.prompts import PromptTemplate
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import os
from uuid import uuid4
import traceback
import logging
import streamlit as st

# --- LangChain Message Types for Chat History ---
from langchain_core.messages import HumanMessage, AIMessage

# ------------------------ LOGGING SETUP ------------------------
log_file = "chatbot_app.log"
log_dir = os.path.dirname(log_file)

# Only create directory if log_dir is not empty (i.e., if log_file specifies a subdirectory)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Log directory '{log_dir}' created.")
elif not log_dir:
    logging.info(f"Logging to current directory. Log file: '{log_file}'")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file), # This will create the file if it doesn't exist
        logging.StreamHandler() # Also log to console
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
# ---------------------------------------------------------------

# ------------------------ CONFIGURATION SETTINGS ------------------------
load_dotenv() # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "Constitution-of-Nepal.pdf"

# --- Qdrant Cloud Configuration (Primary for deployment) ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Local Qdrant Configuration (Fallback for local development if QDRANT_URL is not set) ---
QDRANT_HOST_LOCAL = os.getenv("QDRANT_HOST_LOCAL", "localhost")
QDRANT_PORT_LOCAL = int(os.getenv("QDRANT_PORT_LOCAL", 6333))

QDRANT_COLLECTION_NAME = "nepal_constitution_streamlit" # Updated name

EMBEDDING_MODEL_DIMENSION = 1536
CHAT_MODEL_NAME = "gpt-4o"
# --------------------------------------------------------------------

# == Part 1: Document Processing ==
def extract_text_and_metadata_from_pdf(pdf_filepath):
    logging.info(f"Starting PDF extraction from: {pdf_filepath}")
    if not os.path.exists(pdf_filepath):
        logging.error(f"PDF file not found at: {pdf_filepath}")
        st.error(f"FATAL ERROR: The PDF file '{os.path.basename(pdf_filepath)}' was not found. Ensure it's in the GitHub repository with the script.")
        raise FileNotFoundError(f"The PDF file '{pdf_filepath}' was not found.")
    try:
        pdf_document = fitz.open(pdf_filepath)
    except Exception as e:
        logging.error(f"Could not open PDF '{pdf_filepath}': {e}")
        raise ValueError(f"Failed to open or read the PDF: {pdf_filepath}. Is it a valid PDF?") from e

    extracted_pages = []
    if pdf_document.page_count == 0:
        logging.warning(f"The PDF '{pdf_filepath}' has no pages or is unreadable.")
        raise ValueError("The PDF seems to be empty or unreadable (0 pages).")

    for page_number, page_object in enumerate(pdf_document):
        page_text = page_object.get_text("text", sort=True)
        if page_text.strip():
            page_doc = Document(
                page_content=page_text,
                metadata={
                    "page_number": page_number + 1,
                    "source_document": os.path.basename(pdf_filepath)
                }
            )
            extracted_pages.append(page_doc)
    pdf_document.close()
    if not extracted_pages:
        logging.warning(f"No text could be extracted from the PDF: {pdf_filepath}")
        raise ValueError("No text content was found in the PDF.")
    logging.info(f"Successfully extracted text from {len(extracted_pages)} pages in '{pdf_filepath}'.")
    return extracted_pages

def split_documents_into_smaller_chunks(list_of_documents):
    logging.info("Splitting extracted documents into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    all_chunks = text_splitter.split_documents(list_of_documents)
    logging.info(f"Split documents into {len(all_chunks)} chunks.")
    return all_chunks

# == Part 2: Knowledge Base Creation (Vector Store with Qdrant) ==
def setup_qdrant_vector_store(text_chunks, user_openai_api_key):
    logging.info(f"Setting up Qdrant vector store. Collection name: '{QDRANT_COLLECTION_NAME}'")
    if not user_openai_api_key:
        raise ValueError("OpenAI API key is missing. Cannot create text embeddings.")

    embedding_model = OpenAIEmbeddings(openai_api_key=user_openai_api_key)
    logging.info(f"Using OpenAI embedding model: '{embedding_model.model}' (Dim: {EMBEDDING_MODEL_DIMENSION})")

    qdrant_db_client = None
    if QDRANT_URL: # Prioritize Qdrant Cloud
        logging.info(f"Attempting to connect to Qdrant Cloud at: {QDRANT_URL}")
        if not QDRANT_API_KEY:
            logging.error("QDRANT_URL is set, but QDRANT_API_KEY is missing for Qdrant Cloud.")
            raise ValueError("QDRANT_API_KEY is required when using QDRANT_URL for Qdrant Cloud.")
        try:
            qdrant_db_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
            qdrant_db_client.get_collections() # Test connection
            logging.info(f"Successfully connected to Qdrant Cloud.")
        except Exception as e:
            logging.error(f"Failed to connect to Qdrant Cloud: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to Qdrant Cloud at {QDRANT_URL}. Details: {e}")
    else: # Fallback to local Qdrant
        logging.info(f"QDRANT_URL not set. Attempting to connect to local Qdrant at {QDRANT_HOST_LOCAL}:{QDRANT_PORT_LOCAL}.")
        try:
            qdrant_db_client = QdrantClient(host=QDRANT_HOST_LOCAL, port=QDRANT_PORT_LOCAL, timeout=20)
            qdrant_db_client.get_collections()
            logging.info(f"Successfully connected to local Qdrant server.")
        except ConnectionRefusedError:
            logging.error(f"Local Qdrant connection refused at {QDRANT_HOST_LOCAL}:{QDRANT_PORT_LOCAL}. Is server running?")
            raise ConnectionError(f"Could not connect to local Qdrant. Ensure it's running.")
        except Exception as e:
            logging.error(f"Failed to connect to local Qdrant: {e}", exc_info=True)
            raise ConnectionError(f"Could not connect to local Qdrant. Details: {e}")

    if not qdrant_db_client:
        raise RuntimeError("Qdrant client could not be initialized. Check configuration.")

    collection_is_ready_to_use = False
    try:
        collection_details = qdrant_db_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        current_vector_params = collection_details.config.params.vectors
        if isinstance(current_vector_params, dict) and "" in current_vector_params:
            current_vector_params = current_vector_params[""]

        if current_vector_params and current_vector_params.size == EMBEDDING_MODEL_DIMENSION and \
           current_vector_params.distance == Distance.COSINE:
            num_points_in_collection = qdrant_db_client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count
            if num_points_in_collection > 0 and (not text_chunks or num_points_in_collection >= (len(text_chunks) * 0.8)):
                logging.info(f"Collection '{QDRANT_COLLECTION_NAME}' exists, valid, and populated ({num_points_in_collection} points). Reusing.")
                collection_is_ready_to_use = True
            elif num_points_in_collection > 0:
                logging.warning(f"Collection '{QDRANT_COLLECTION_NAME}' exists, valid, but point count ({num_points_in_collection}) differs from expected ({len(text_chunks) if text_chunks else 0}). Reusing existing data.")
                collection_is_ready_to_use = True
            else:
                logging.info(f"Collection '{QDRANT_COLLECTION_NAME}' exists, valid, but empty. Will populate.")
        else:
            actual_size = current_vector_params.size if current_vector_params else "N/A"
            actual_distance = current_vector_params.distance if current_vector_params else "N/A"
            logging.warning(f"Collection '{QDRANT_COLLECTION_NAME}' exists but MISMATCHED. Expected: Dim={EMBEDDING_MODEL_DIMENSION}, Dist={Distance.COSINE}. Found: Dim={actual_size}, Dist={actual_distance}. RECREATING.")
            qdrant_db_client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_MODEL_DIMENSION, distance=Distance.COSINE)
            )
    except (qdrant_exceptions.UnexpectedResponse, ValueError) as e:
        if "not found" in str(e).lower() or (isinstance(e, qdrant_exceptions.UnexpectedResponse) and e.status_code == 404):
            logging.info(f"Collection '{QDRANT_COLLECTION_NAME}' not found. Creating new collection.")
            qdrant_db_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_MODEL_DIMENSION, distance=Distance.COSINE)
            )
        else:
            logging.error(f"Error checking Qdrant collection '{QDRANT_COLLECTION_NAME}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to setup Qdrant collection. Details: {e}")
    except AttributeError:
        logging.warning(f"AttributeError verifying collection '{QDRANT_COLLECTION_NAME}'. Recreating.")
        try:
            qdrant_db_client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_MODEL_DIMENSION, distance=Distance.COSINE))
            collection_is_ready_to_use = False
        except Exception as recreate_e:
            logging.error(f"Failed to recreate collection after AttributeError: {recreate_e}")
            raise RuntimeError(f"Critical error setting up Qdrant. Details: {recreate_e}")
    except Exception as e:
        logging.error(f"Unexpected error during Qdrant collection setup for '{QDRANT_COLLECTION_NAME}': {e}", exc_info=True)
        raise RuntimeError(f"Critical error setting up Qdrant. Details: {e}")


    if not collection_is_ready_to_use:
        if not text_chunks:
            logging.warning("No text chunks from PDF. Vector store may be empty or use stale data.")
        else:
            logging.info(f"Populating Qdrant collection '{QDRANT_COLLECTION_NAME}'. This may take time/cost.")
            chunk_page_contents = [chunk.page_content for chunk in text_chunks]
            try:
                chunk_vectors = embedding_model.embed_documents(chunk_page_contents)
            except (APIError, AuthenticationError, APIConnectionError) as e:
                raise RuntimeError(f"OpenAI API error during embedding: {e}")
            except Exception as e:
                raise RuntimeError(f"Embedding process failed: {e}")

            if chunk_vectors and len(chunk_vectors[0]) != EMBEDDING_MODEL_DIMENSION:
                raise RuntimeError("Embedding dimension mismatch!")

            points_to_add = [
                PointStruct(id=str(uuid4()), vector=vec, payload={"page_content": chunk.page_content, "metadata": chunk.metadata})
                for vec, chunk in zip(chunk_vectors, text_chunks)
            ]
            batch_size = 100
            try:
                for i in range(0, len(points_to_add), batch_size):
                    qdrant_db_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points_to_add[i:i+batch_size], wait=True)
                    logging.info(f"Upserted batch {i//batch_size + 1}/{(len(points_to_add)-1)//batch_size + 1}.")
                logging.info("All points added to Qdrant.")
            except Exception as e:
                raise RuntimeError(f"Qdrant upsert failed: {e}")
    else:
        logging.info(f"Skipped populating. Using existing data in '{QDRANT_COLLECTION_NAME}'.")

    langchain_qdrant_vector_store = Qdrant(
        client=qdrant_db_client,
        collection_name=QDRANT_COLLECTION_NAME,
        embeddings=embedding_model
    )
    logging.info(f"LangChain Qdrant vector store adapter for '{QDRANT_COLLECTION_NAME}' ready.")
    return langchain_qdrant_vector_store

# == Part 3: Conversational AI Setup == (Identical to previous good version)
def create_conversational_ai_chain(vector_store_for_retrieval, user_openai_api_key):
    logging.info("Setting up the conversational AI chain...")
    if not user_openai_api_key:
        raise ValueError("OpenAI API key is missing for AI chat model.")

    llm_chat_model = ChatOpenAI(
        openai_api_key=user_openai_api_key,
        model_name=CHAT_MODEL_NAME,
        temperature=0.2
    )
    condense_question_template = """
    Given the conversation history below and a follow-up question,
    rephrase the follow-up question so it's a complete, standalone question.
    This standalone question should be in the same language as the follow-up question.

    You are an AI assistant focused ONLY on the Constitution of Nepal.
    If the follow-up question is clearly NOT about the Constitution of Nepal
    (e.g., asking for recipes, weather, or laws of other countries not mentioned in the Nepal Constitution),
    rephrase it to indicate it's off-topic. For example:
    "The user is asking about [the off-topic subject], which is outside the scope of the Constitution of Nepal."

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    answer_generation_template = """
    You are an AI assistant specializing in the Constitution of Nepal.
    Your task is to answer the given question based ONLY on the "Context" provided below.
    The "Context" consists of excerpts from the Constitution of Nepal.

    IMPORTANT RULES:
    1. Your answer MUST be derived solely from the information within the provided "Context".
    2. Do NOT use any external knowledge or make up information.
    3. If the "Context" does not contain the information to answer the question, you MUST state:
       "The provided document (Constitution of Nepal) does not contain information to answer this question."
    4. If the "Question" itself (after being rephrased) indicates it's off-topic for the Constitution of Nepal
       (e.g., "The user is asking about cooking recipes..."), then your answer should be:
       "This question is about a topic outside the scope of the Constitution of Nepal."
    5. If providing lists or multiple points from the context, format them clearly using markdown bullet points or numbered lists.
    6. Be concise and directly answer the question based on the context.

    Context:
    {context}

    Question: {question}
    Helpful Answer:"""
    ANSWER_GENERATION_PROMPT = PromptTemplate(template=answer_generation_template, input_variables=["context", "question"])

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_chat_model,
        retriever=vector_store_for_retrieval.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": ANSWER_GENERATION_PROMPT},
        return_source_documents=True,
    )
    logging.info("Conversational AI chain created successfully.")
    return conversational_chain

# == Part 4: Interacting with the Chatbot == (Identical to previous good version)
def get_answer_from_ai(ai_chain, user_question, conversation_history):
    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("Question must be a non-empty string.")
    logging.info(f"Sending question to AI: '{user_question}' (History: {len(conversation_history)//2} turns)")
    try:
        return ai_chain.invoke({"question": user_question, "chat_history": conversation_history})
    except (APIError, AuthenticationError, APIConnectionError) as e:
        logging.error(f"OpenAI API error: {e}")
        if isinstance(e, AuthenticationError):
            st.error("OpenAI Authentication Failed. Check API key configuration.")
            raise RuntimeError("OpenAI Authentication Failed.")
        raise RuntimeError(f"OpenAI API issue: {e}")
    except Exception as e:
        logging.error(f"Unexpected error from AI chain: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error getting answer: {e}")

# == Part 5: Streamlit Web Application UI ==
@st.cache_resource(show_spinner="Setting up Constitution Chatbot...")
def initialize_chatbot_backend():
    logging.info("Streamlit: Initializing chatbot backend...")
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API Key not configured. Set in .env or Streamlit secrets.")

    # Check for Qdrant Cloud vars IF QDRANT_URL is set (primary deployment mode)
    # Or if running in Streamlit Cloud and QDRANT_URL is expected but missing
    is_streamlit_cloud = os.getenv("STREAMLIT_SERVER_MODE") == "cloud" or "streamlit.app" in os.getenv("STREAMLIT_SERVER_ORIGIN", "")

    if QDRANT_URL and not QDRANT_API_KEY: # If URL is set, API key must also be set
         raise ValueError("QDRANT_API_KEY not configured for Qdrant Cloud. Set in .env or Streamlit secrets.")
    if is_streamlit_cloud and not QDRANT_URL: # In cloud, QDRANT_URL must be set
         raise ValueError("QDRANT_URL not configured for cloud deployment. Set in Streamlit secrets.")


    document_pages = extract_text_and_metadata_from_pdf(PDF_PATH)
    text_chunks_from_pdf = split_documents_into_smaller_chunks(document_pages)
    qdrant_knowledge_base = setup_qdrant_vector_store(text_chunks_from_pdf, OPENAI_API_KEY)
    fully_initialized_ai_chain = create_conversational_ai_chain(qdrant_knowledge_base, OPENAI_API_KEY)
    logging.info("Streamlit: Chatbot backend initialized successfully.")
    return fully_initialized_ai_chain

def run_constitution_chatbot_app():
    st.set_page_config(page_title="🇳🇵 Constitution of Nepal Chatbot", page_icon="🇳🇵", layout="wide")
    st.title("🇳🇵 Chat with the Constitution of Nepal")
    st.caption(f"AI Model: '{CHAT_MODEL_NAME}'. Source: '{os.path.basename(PDF_PATH)}'")
    # st.markdown(f"Logs: `{log_file}` (local) / deployment console.") # Optional dev info
    st.markdown("---")

    # Pre-initialization checks for clearer UI errors
    if not OPENAI_API_KEY:
        st.error("CRITICAL: OpenAI API Key is missing. Please configure it.")
        return
    is_streamlit_cloud_ui = os.getenv("STREAMLIT_SERVER_MODE") == "cloud" or "streamlit.app" in os.getenv("STREAMLIT_SERVER_ORIGIN", "")
    if is_streamlit_cloud_ui:
        if not QDRANT_URL:
            st.error("CRITICAL: QDRANT_URL (for Qdrant Cloud) is not set in Streamlit Secrets. Please configure it.")
            return
        if not QDRANT_API_KEY:
            st.error("CRITICAL: QDRANT_API_KEY (for Qdrant Cloud) is not set in Streamlit Secrets. Please configure it.")
            return

    try:
        ai_conversation_handler = initialize_chatbot_backend()
    except Exception as e:
        st.error(f"**FATAL ERROR DURING CHATBOT INITIALIZATION:**\n\n`{str(e)}`\n\nCheck logs for details. Ensure PDF, API keys, and Qdrant (local/cloud) are correctly set up.")
        logging.critical(f"Streamlit UI: Backend initialization failed: {e}", exc_info=True)
        return

    # Session state initialization (same as before)
    if "chat_messages_for_display" not in st.session_state:
        st.session_state.chat_messages_for_display = [{"role": "assistant", "content": "Namaste! Ask about the Constitution of Nepal."}]
    if "langchain_chat_history_objects" not in st.session_state:
        st.session_state.langchain_chat_history_objects = []
    if "last_ai_response_details" not in st.session_state:
        st.session_state.last_ai_response_details = None
    if "global_feedback_status_for_last_response" not in st.session_state:
        st.session_state.global_feedback_status_for_last_response = None
    if "global_feedback_target_id" not in st.session_state:
        st.session_state.global_feedback_target_id = None

    # Chat display (same as before)
    chat_display_area = st.container()
    with chat_display_area:
        for msg_data in st.session_state.chat_messages_for_display:
            with st.chat_message(msg_data["role"]):
                st.markdown(msg_data["content"])
                if msg_data["role"] == "assistant" and msg_data.get("sources"):
                    with st.expander("View Sources", expanded=False):
                        for src_info in msg_data["sources"]:
                            st.markdown(f"**Source:** Page {src_info['page_number']} ({src_info['document_name']})\n> _{src_info['preview_text']}..._")

    user_query = st.chat_input("Ask your question...")

    # Feedback buttons 
    if st.session_state.last_ai_response_details:
        st.markdown("---")
        last_resp_id = st.session_state.last_ai_response_details["id"]

        # MODIFIED LINE: Ensure feedback_given is explicitly boolean
        feedback_given_for_current_response = (
            st.session_state.global_feedback_status_for_last_response is not None and
            st.session_state.global_feedback_target_id == last_resp_id
        )
        # Or even more explicitly:
        # feedback_given_for_current_response = bool(
        #     st.session_state.global_feedback_status_for_last_response and # Check if not None and not empty string
        #     st.session_state.global_feedback_target_id == last_resp_id
        # )

        cols = st.columns([0.08, 0.08, 0.84])

        # Use the new boolean variable for button type and disabled state
        up_type = "primary" if st.session_state.global_feedback_status_for_last_response == "liked" and feedback_given_for_current_response else "secondary"
        down_type = "primary" if st.session_state.global_feedback_status_for_last_response == "disliked" and feedback_given_for_current_response else "secondary"

        if cols[0].button("👍", key=f"up_{last_resp_id}",
                          disabled=feedback_given_for_current_response, # Use the explicitly boolean variable
                          type=up_type, use_container_width=True, help="Helpful"):
            if not feedback_given_for_current_response: # Check the boolean variable
                st.session_state.global_feedback_status_for_last_response = "liked"
                st.session_state.global_feedback_target_id = last_resp_id
                st.toast("Thanks for feedback! 😊", icon="😊")
                logging.info(f"Feedback: Liked ID {last_resp_id}")
                st.rerun()
        if cols[1].button("👎", key=f"down_{last_resp_id}",
                          disabled=feedback_given_for_current_response, # Use the explicitly boolean variable
                          type=down_type, use_container_width=True, help="Not helpful"):
            if not feedback_given_for_current_response: # Check the boolean variable
                st.session_state.global_feedback_status_for_last_response = "disliked"
                st.session_state.global_feedback_target_id = last_resp_id
                st.toast("Thanks for feedback. We'll improve! 😕", icon="😕")
                logging.info(f"Feedback: Disliked ID {last_resp_id}")
                st.rerun()

    # Process user input (same as before)
    if user_query:
        st.session_state.chat_messages_for_display.append({"role": "user", "content": user_query})
        st.session_state.langchain_chat_history_objects.append(HumanMessage(content=user_query))
        st.session_state.global_feedback_status_for_last_response = None # Reset for new response
        st.session_state.global_feedback_target_id = None

        ai_answer, sources_display = "", []
        with st.spinner("Thinking..."):
            try:
                response = get_answer_from_ai(ai_conversation_handler, user_query, st.session_state.langchain_chat_history_objects)
                ai_answer = response.get('answer', "Error generating answer.")
                src_docs = response.get('source_documents', [])
                if src_docs:
                    unique_srcs = {}
                    for doc in src_docs:
                        key = (doc.metadata.get('page_number', 'N/A'), doc.page_content[:50])
                        if key not in unique_srcs:
                            sources_display.append({
                                "page_number": doc.metadata.get('page_number', 'N/A'),
                                "document_name": doc.metadata.get('source_document', 'Unknown'),
                                "preview_text": doc.page_content[:150].strip().replace("\n", " ")
                            })
                            unique_srcs[key] = True
            except Exception as e:
                ai_answer = f"Error: {e}"
                logging.error(f"Error processing query: {e}", exc_info=True)

        assistant_msg = {"role": "assistant", "content": ai_answer}
        if sources_display: assistant_msg["sources"] = sources_display
        st.session_state.chat_messages_for_display.append(assistant_msg)
        st.session_state.langchain_chat_history_objects.append(AIMessage(content=ai_answer))

        new_resp_id = f"ai_resp_{len(st.session_state.chat_messages_for_display)}"
        st.session_state.last_ai_response_details = {"content": ai_answer, "id": new_resp_id}
        st.session_state.global_feedback_target_id = new_resp_id

        MAX_TURNS = 7
        if len(st.session_state.langchain_chat_history_objects) > MAX_TURNS * 2:
            st.session_state.langchain_chat_history_objects = st.session_state.langchain_chat_history_objects[-(MAX_TURNS*2):]
            logging.info(f"History trimmed to last {MAX_TURNS} turns.")
        st.rerun()

# --- Main execution point ---
if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    print(f"To run this Streamlit application ({script_name}):")
    print(f"1. Ensure '{PDF_PATH}' is present.")
    print("2. For Cloud Qdrant: Set OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY in .env or environment.")
    print("3. For Local Qdrant: Set OPENAI_API_KEY in .env. Ensure QDRANT_URL is NOT set. Optionally set QDRANT_HOST_LOCAL/QDRANT_PORT_LOCAL if not defaults.")
    print(f"4. Run: streamlit run {script_name}")
    run_constitution_chatbot_app()