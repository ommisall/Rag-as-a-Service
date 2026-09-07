import streamlit as st
import requests
from anthropic import Anthropic
import time
import json
from typing import List, Dict, Optional, Generator, Any
from urllib.parse import urlparse

# ==============================================================================
# RAG Pipeline Service
# ==============================================================================
class RAGPipeline:
    def __init__(self, ragie_api_key: str, anthropic_api_key: str):
        """
        Initialize the RAG pipeline with Ragie and Anthropic API keys.
        """
        self.ragie_api_key = ragie_api_key.strip()
        self.anthropic_api_key = anthropic_api_key.strip()
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
        
        # Ragie Endpoints
        self.BASE_URL = "https://api.ragie.ai"
        self.RAGIE_DOCS_URL = f"{self.BASE_URL}/documents"
        self.RAGIE_UPLOAD_URL = f"{self.BASE_URL}/documents/url"
        self.RAGIE_RAW_URL = f"{self.BASE_URL}/documents/raw"
        self.RAGIE_RETRIEVAL_URL = f"{self.BASE_URL}/retrievals"

    @property
    def auth_headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "authorization": f"Bearer {self.ragie_api_key}"
        }

    def upload_url(self, url: str, name: Optional[str] = None, mode: str = "fast", scope: Optional[str] = None) -> Dict:
        """Upload a document to Ragie from a URL."""
        if not name:
            parsed = urlparse(url)
            name = parsed.path.split('/')[-1] or parsed.netloc or "web_document"
            
        payload: Dict[str, Any] = {
            "mode": mode,
            "name": name,
            "url": url
        }
        if scope:
            payload["metadata"] = {"scope": scope}
        
        headers = {**self.auth_headers, "content-type": "application/json"}
        response = requests.post(self.RAGIE_UPLOAD_URL, json=payload, headers=headers)
        
        if not response.ok:
            raise Exception(f"URL upload failed ({response.status_code}): {response.text}")
        return response.json()

    def upload_file(self, file_bytes: bytes, filename: str, mode: str = "fast", scope: Optional[str] = None) -> Dict:
        """Upload a local file (PDF, TXT, MD, etc.) directly to Ragie."""
        files = {
            "file": (filename, file_bytes)
        }
        data: Dict[str, Any] = {
            "mode": mode,
            "name": filename
        }
        if scope:
            data["metadata"] = json.dumps({"scope": scope})
            
        # Requests automatically sets multipart/form-data boundary with files param
        response = requests.post(
            self.RAGIE_DOCS_URL,
            files=files,
            data=data,
            headers={"authorization": f"Bearer {self.ragie_api_key}"}
        )
        if not response.ok:
            raise Exception(f"File upload failed ({response.status_code}): {response.text}")
        return response.json()

    def upload_raw_text(self, text: str, name: str = "Pasted Note", mode: str = "fast", scope: Optional[str] = None) -> Dict:
        """Upload raw pasted text or notes to Ragie."""
        payload: Dict[str, Any] = {
            "data": text,
            "name": name,
            "mode": mode
        }
        if scope:
            payload["metadata"] = {"scope": scope}
            
        headers = {**self.auth_headers, "content-type": "application/json"}
        response = requests.post(self.RAGIE_RAW_URL, json=payload, headers=headers)
        if not response.ok:
            raise Exception(f"Raw text upload failed ({response.status_code}): {response.text}")
        return response.json()

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents in Ragie."""
        response = requests.get(self.RAGIE_DOCS_URL, headers=self.auth_headers)
        if not response.ok:
            raise Exception(f"Failed to list documents ({response.status_code}): {response.text}")
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("documents") or data.get("items") or []
        return []

    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID from Ragie."""
        url = f"{self.RAGIE_DOCS_URL}/{document_id}"
        response = requests.delete(url, headers=self.auth_headers)
        if not response.ok:
            raise Exception(f"Failed to delete document ({response.status_code}): {response.text}")
        return True

    def retrieve_chunks(self, query: str, top_k: int = 5, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant scored chunks from Ragie."""
        headers = {**self.auth_headers, "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k
        }
        if scope and scope.strip():
            payload["filters"] = {"scope": scope.strip()}

        response = requests.post(self.RAGIE_RETRIEVAL_URL, headers=headers, json=payload)
        if not response.ok:
            raise Exception(f"Retrieval failed ({response.status_code}): {response.text}")
            
        data = response.json()
        scored_chunks = data.get("scored_chunks", [])
        return scored_chunks

    def create_system_prompt(self, chunk_data: List[Dict[str, Any]], persona: str = "grounded") -> str:
        """Create an intelligent system prompt embedding retrieved chunks and selected persona."""
        formatted_chunks = []
        for i, chunk in enumerate(chunk_data, 1):
            text = chunk.get("text", "").strip()
            score = chunk.get("score")
            doc_name = chunk.get("document_name", "Document")
            score_str = f" [Score: {score:.3f}]" if score is not None else ""
            formatted_chunks.append(f"--- Chunk {i} (Source: {doc_name}{score_str}) ---\n{text}")

        context_block = "\n\n".join(formatted_chunks) if formatted_chunks else "No relevant context found in documents."

        persona_instructions = {
            "grounded": (
                "You are an accurate, honest RAG assistant. Answer strictly based on the retrieved context below. "
                "If the information is not contained in the context, clearly state that the provided documents do not contain this information. "
                "Do not hallucinate or speculate."
            ),
            "detailed": (
                "You are an expert analytical research assistant. Provide an in-depth, structured, and comprehensive explanation "
                "drawing from all nuances in the retrieved context below. Use headers, bullet points, and code blocks where helpful."
            ),
            "executive": (
                "You are an executive briefing assistant. Deliver concise, high-level summaries, key bulleted takeaways, "
                "and actionable insights directly addressing the user's inquiry from the context below."
            )
        }
        chosen_persona = persona_instructions.get(persona, persona_instructions["grounded"])

        return f"""{chosen_persona}

### Retrieved Context:
{context_block}

### Guidelines:
- Answer informally, directly, and concisely using Markdown formatting.
- If referencing specific data points, cite the source chunk or document when helpful.
- If the context has no relevant information for the user's question, let the user know politely.
"""

    def stream_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.5,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream response tokens from Claude via Anthropic API."""
        with self.anthropic_client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

# ==============================================================================
# UI Helper & State Management
# ==============================================================================
def initialize_session_state():
    """Ensure all required session states exist."""
    defaults = {
        "pipeline": None,
        "api_keys_submitted": False,
        "messages": [],
        "documents_cache": None,
        "last_refresh_time": 0,
        "ragie_key": "",
        "anthropic_key": ""
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def refresh_documents(pipeline: RAGPipeline):
    """Fetch current documents from Ragie."""
    try:
        docs = pipeline.list_documents()
        st.session_state.documents_cache = docs
        st.session_state.last_refresh_time = time.time()
    except Exception as e:
        st.sidebar.error(f"Could not fetch documents: {e}")

# ==============================================================================
# Main Application
# ==============================================================================
def main():
    st.set_page_config(
        page_title="RAG-as-a-Service Studio",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    initialize_session_state()

    # Custom styling
    st.markdown("""
    <style>
    .source-chip {
        display: inline-block;
        background: #f0f2f6;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid #e0e4eb;
    }
    .status-badge {
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
    }
    .badge-ready { background-color: #d1fae5; color: #065f46; }
    .badge-indexing { background-color: #fef3c7; color: #92400e; }
    .badge-failed { background-color: #fee2e2; color: #991b1b; }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SIDEBAR: Configuration, Model Settings, & Document Library
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🔑 API Authentication")
        ragie_input = st.text_input(
            "Ragie API Key",
            type="password",
            value=st.session_state.ragie_key,
            help="Get from https://ragie.ai"
        )
        anthropic_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.anthropic_key,
            help="Get from https://console.anthropic.com"
        )

        if st.button("Save API Keys", use_container_width=True, type="primary"):
            if ragie_input and anthropic_input:
                try:
                    pipeline = RAGPipeline(ragie_input, anthropic_input)
                    st.session_state.pipeline = pipeline
                    st.session_state.ragie_key = ragie_input
                    st.session_state.anthropic_key = anthropic_input
                    st.session_state.api_keys_submitted = True
                    refresh_documents(pipeline)
                    st.success("API keys connected!")
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")
            else:
                st.warning("Please enter both keys.")

        if st.session_state.api_keys_submitted:
            st.success("🟢 Connected to Ragie & Claude", icon="✅")
        else:
            st.info("Please enter and save your API keys to begin.")

        st.divider()

        # Model & Generation Settings
        st.markdown("### ⚙️ Model & Retrieval")
        model_choice = st.selectbox(
            "Claude Model",
            options=[
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229"
            ],
            index=0
        )
        
        persona = st.selectbox(
            "Assistant Persona",
            options=["grounded", "detailed", "executive"],
            format_func=lambda x: {
                "grounded": "🎯 Grounded (Strict Facts Only)",
                "detailed": "🔬 Detailed & Explanatory",
                "executive": "📊 Executive Briefing"
            }[x]
        )

        top_k = st.slider("Top Chunks (k)", min_value=1, max_value=15, value=5, step=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        scope_filter = st.text_input("Scope Filter (optional)", value="", placeholder="e.g. tutorial or user_id")

        st.divider()

        # Document Library Inspector in Sidebar
        st.markdown("### 📚 Ingested Documents")
        if st.session_state.pipeline:
            col_ref, col_count = st.columns([1, 1])
            with col_ref:
                if st.button("🔄 Refresh", use_container_width=True):
                    refresh_documents(st.session_state.pipeline)

            docs = st.session_state.documents_cache or []
            with col_count:
                st.caption(f"Total: **{len(docs)}**")

            if docs:
                for doc in docs[:10]:  # Show top 10
                    doc_id = doc.get("id") or doc.get("document_id", "")
                    doc_name = doc.get("name") or doc.get("document_name") or "Untitled"
                    status = doc.get("status", "ready").lower()
                    
                    status_class = "badge-ready" if status in ["ready", "indexed"] else ("badge-indexing" if status in ["indexing", "processing"] else "badge-failed")

                    with st.expander(f"📄 {doc_name[:22]}"):
                        st.markdown(f"**Status**: <span class='status-badge {status_class}'>{status.upper()}</span>", unsafe_allow_html=True)
                        st.caption(f"ID: `{doc_id[:16]}...`")
                        if doc.get("chunk_count"):
                            st.caption(f"Chunks: {doc.get('chunk_count')}")
                        if st.button("🗑️ Delete", key=f"del_{doc_id}", use_container_width=True):
                            try:
                                st.session_state.pipeline.delete_document(doc_id)
                                st.success("Deleted!")
                                refresh_documents(st.session_state.pipeline)
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Error: {ex}")
            else:
                st.caption("No documents indexed yet.")
        else:
            st.caption("Connect API keys to view indexed documents.")

    # --------------------------------------------------------------------------
    # MAIN WORKSPACE: Header & Tabs
    # --------------------------------------------------------------------------
    st.title("⚡ RAG-as-a-Service Studio")
    st.caption("Intelligent Retrieval-Augmented Generation powered by Ragie AI & Claude 3.5")

    tab_chat, tab_ingest = st.tabs(["💬 Chat with Documents", "📥 Ingest Documents"])

    # --------------------------------------------------------------------------
    # TAB 1: Chat Interface
    # --------------------------------------------------------------------------
    with tab_chat:
        col_actions, col_dl = st.columns([6, 2])
        with col_dl:
            # Download conversation as Markdown
            if st.session_state.messages:
                chat_export = "# RAG Chat Export\n\n"
                for m in st.session_state.messages:
                    chat_export += f"### {m['role'].capitalize()}\n{m['content']}\n\n"
                st.download_button(
                    label="💾 Export Chat (MD)",
                    data=chat_export,
                    file_name="rag_chat_export.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        with col_actions:
            if st.session_state.messages and st.button("🧹 Clear Conversation", type="secondary"):
                st.session_state.messages = []
                st.rerun()

        # Render message history
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            sources = msg.get("sources", [])

            with st.chat_message(role):
                st.markdown(content)
                if sources:
                    with st.expander(f"🔍 Source Citations ({len(sources)} chunks retrieved)"):
                        for idx, s in enumerate(sources, 1):
                            score_text = f" • Similarity Score: `{s.get('score', 0):.4f}`" if s.get('score') is not None else ""
                            doc_title = s.get("document_name") or "Document"
                            st.markdown(f"**Chunk {idx}: {doc_title}**{score_text}")
                            st.info(s.get("text", ""))

        # Starter prompts if history is empty
        if not st.session_state.messages:
            st.markdown("##### 💡 Starter Prompts")
            prompt_cols = st.columns(3)
            starters = [
                "Summarize the key takeaways from the documents",
                "What are the main technical requirements mentioned?",
                "Extract all important dates, entities, or action items"
            ]
            for i, p_text in enumerate(starters):
                if prompt_cols[i].button(p_text, use_container_width=True):
                    st.session_state["preset_prompt"] = p_text
                    st.rerun()

        # Check preset prompt or new chat input
        preset = st.session_state.pop("preset_prompt", None)
        user_input = st.chat_input("Ask a question about your indexed documents...")
        prompt = preset or user_input

        if prompt:
            if not st.session_state.api_keys_submitted or not st.session_state.pipeline:
                st.error("⚠️ Please configure your Ragie and Anthropic API keys in the sidebar first.")
            else:
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate Assistant response with RAG
                with st.chat_message("assistant"):
                    pipeline: RAGPipeline = st.session_state.pipeline
                    with st.spinner("Retrieving relevant knowledge chunks..."):
                        try:
                            retrieved_chunks = pipeline.retrieve_chunks(
                                query=prompt,
                                top_k=top_k,
                                scope=scope_filter if scope_filter.strip() else None
                            )
                        except Exception as e:
                            retrieved_chunks = []
                            st.warning(f"Note on retrieval: {e}")

                    # Build conversation messages for Anthropic
                    anthropic_msgs = []
                    for m in st.session_state.messages:
                        if m["role"] in ["user", "assistant"]:
                            anthropic_msgs.append({"role": m["role"], "content": m["content"]})

                    system_prompt = pipeline.create_system_prompt(retrieved_chunks, persona=persona)

                    # Stream response
                    try:
                        stream_gen = pipeline.stream_response(
                            system_prompt=system_prompt,
                            messages=anthropic_msgs,
                            model=model_choice,
                            temperature=temperature
                        )
                        full_response = st.write_stream(stream_gen)

                        # Display source citations accordion
                        if retrieved_chunks:
                            with st.expander(f"🔍 Source Citations ({len(retrieved_chunks)} chunks retrieved)"):
                                for idx, s in enumerate(retrieved_chunks, 1):
                                    score_text = f" • Similarity Score: `{s.get('score', 0):.4f}`" if s.get('score') is not None else ""
                                    doc_title = s.get("document_name") or "Document"
                                    st.markdown(f"**Chunk {idx}: {doc_title}**{score_text}")
                                    st.info(s.get("text", ""))

                        # Append to state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": retrieved_chunks
                        })
                    except Exception as e:
                        st.error(f"Generation error: {e}")

    # --------------------------------------------------------------------------
    # TAB 2: Multi-source Document Ingestion
    # --------------------------------------------------------------------------
    with tab_ingest:
        if not st.session_state.api_keys_submitted or not st.session_state.pipeline:
            st.info("👉 Enter your Ragie and Anthropic API keys in the sidebar to start uploading documents.")
        else:
            st.markdown("### 📥 Ingest Documents into Ragie")
            st.caption("Upload documents to automatically extract, chunk, and embed them for retrieval.")

            ingest_type = st.radio(
                "Select Ingestion Method",
                ["📄 File Upload (PDF, TXT, MD, DOCX)", "🌐 Web URL", "✍️ Raw Text / Notes"],
                horizontal=True
            )

            col_mode, col_scope = st.columns(2)
            with col_mode:
                upload_mode = st.selectbox(
                    "Parsing Mode",
                    ["fast", "accurate"],
                    help="'fast' uses standard heuristics; 'accurate' employs vision/OCR models for complex layouts."
                )
            with col_scope:
                custom_scope = st.text_input("Assign Scope / Partition (optional)", placeholder="e.g. tutorial or finance")

            st.divider()

            pipeline: RAGPipeline = st.session_state.pipeline

            # Option A: File Upload
            if "File Upload" in ingest_type:
                uploaded_files = st.file_uploader(
                    "Choose one or more files",
                    accept_multiple_files=True,
                    type=["pdf", "txt", "md", "docx", "csv", "json"]
                )
                if uploaded_files and st.button("🚀 Upload & Index Files", type="primary"):
                    success_count = 0
                    progress_bar = st.progress(0)
                    for idx, up_file in enumerate(uploaded_files):
                        file_bytes = up_file.getvalue()
                        with st.spinner(f"Ingesting {up_file.name}..."):
                            try:
                                pipeline.upload_file(
                                    file_bytes=file_bytes,
                                    filename=up_file.name,
                                    mode=upload_mode,
                                    scope=custom_scope if custom_scope.strip() else None
                                )
                                success_count += 1
                            except Exception as ex:
                                st.error(f"Failed to upload {up_file.name}: {ex}")
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    if success_count > 0:
                        st.success(f"Successfully uploaded {success_count} file(s)! Indexing has begun.")
                        time.sleep(2)
                        refresh_documents(pipeline)
                        st.rerun()

            # Option B: URL Upload
            elif "Web URL" in ingest_type:
                url_input = st.text_input("Enter Document or Webpage URL", placeholder="https://example.com/handbook.pdf")
                doc_title = st.text_input("Custom Document Name (optional)")

                if st.button("🚀 Ingest from URL", type="primary"):
                    if url_input:
                        with st.spinner(f"Submitting URL for indexing..."):
                            try:
                                res = pipeline.upload_url(
                                    url=url_input,
                                    name=doc_title if doc_title.strip() else None,
                                    mode=upload_mode,
                                    scope=custom_scope if custom_scope.strip() else None
                                )
                                st.success("URL submitted successfully! Indexing in progress.")
                                time.sleep(2)
                                refresh_documents(pipeline)
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Error: {ex}")
                    else:
                        st.warning("Please provide a valid URL.")

            # Option C: Raw Text Paste
            else:
                raw_title = st.text_input("Note / Document Title", value="Quick Note")
                raw_content = st.text_area("Paste Content / Transcript / Markdown", height=220)

                if st.button("🚀 Ingest Raw Text", type="primary"):
                    if raw_content.strip():
                        with st.spinner("Submitting raw text..."):
                            try:
                                pipeline.upload_raw_text(
                                    text=raw_content,
                                    name=raw_title,
                                    mode=upload_mode,
                                    scope=custom_scope if custom_scope.strip() else None
                                )
                                st.success("Text uploaded and queued for indexing!")
                                time.sleep(2)
                                refresh_documents(pipeline)
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Error: {ex}")
                    else:
                        st.warning("Please paste some text before submitting.")

if __name__ == "__main__":
    main()