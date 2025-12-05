"""
Sidebar UI Components.

Retrieval settings and configuration components for the sidebar.
"""

import streamlit as st
from typing import Any


def render_retrieval_settings(config: Any):
    """Render the retrieval settings section in the sidebar.

    Args:
        config: The application config object with retrieval settings.
    """
    st.sidebar.markdown("### Retrieval Settings")

    # Retrieval Profile (Presets)
    presets = config.get_retrieval_presets()
    preset_names = list(presets.keys()) + ["custom"]

    # Format function for preset display
    def format_preset(preset_name: str) -> str:
        if preset_name == "custom":
            return "⚙️ Custom"
        preset = presets.get(preset_name, {})
        icon = preset.get("icon", "")
        display = preset.get("display_name", preset_name)
        return f"{icon} {display}"

    selected_preset = st.sidebar.selectbox(
        "Retrieval Profile",
        options=preset_names,
        index=preset_names.index(st.session_state.current_preset)
        if st.session_state.current_preset in preset_names else 0,
        format_func=format_preset,
        help="Pre-configured retrieval settings for different use cases"
    )
    st.session_state.current_preset = selected_preset

    # Apply preset values or show custom controls
    if selected_preset != "custom":
        preset = presets[selected_preset]

        # Show preset description
        st.sidebar.info(f"{preset.get('icon', '')} {preset.get('description', '')}")

        # Apply preset values to session state
        st.session_state.search_k = preset.get("k", 5)
        st.session_state.hybrid_alpha = preset.get("alpha", 0.5)
        st.session_state.use_reranking = preset.get("rerank", True)
        st.session_state.current_retrieval_method = preset.get("method", "hybrid")

        # Show current settings (read-only)
        with st.sidebar.expander("Preset Settings", expanded=False):
            st.markdown(f"**Results**: {st.session_state.search_k}")
            st.markdown(f"**Alpha**: {st.session_state.hybrid_alpha}")
            st.markdown(f"**Re-ranking**: {'On' if st.session_state.use_reranking else 'Off'}")
            st.markdown(f"**Method**: {st.session_state.current_retrieval_method}")

    else:
        # Custom mode - show all controls
        _render_custom_retrieval_controls(config)

    st.sidebar.markdown("---")


def _render_custom_retrieval_controls(config: Any):
    """Render custom retrieval method controls.

    Args:
        config: The application config object.
    """
    retrieval_methods = {
        "Semantic Only": "semantic",
        "BM25 Only": "bm25",
        "Hybrid (BM25 + Semantic)": "hybrid"
    }

    selected_method = st.sidebar.selectbox(
        "Retrieval Method",
        options=list(retrieval_methods.keys()),
        index=list(retrieval_methods.values()).index(
            st.session_state.current_retrieval_method
        ) if st.session_state.current_retrieval_method in retrieval_methods.values() else 2,
        help="Choose retrieval strategy"
    )
    st.session_state.current_retrieval_method = retrieval_methods[selected_method]

    # Alpha slider for hybrid mode
    if st.session_state.current_retrieval_method == "hybrid":
        st.session_state.hybrid_alpha = st.sidebar.slider(
            "Semantic Weight (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=getattr(st.session_state, 'hybrid_alpha', config.get_hybrid_alpha()),
            step=0.1,
            help="0 = BM25 only, 1 = Semantic only"
        )

    # Re-ranking toggle
    st.session_state.use_reranking = st.sidebar.checkbox(
        "Enable Re-ranking",
        value=getattr(st.session_state, 'use_reranking', config.is_reranking_enabled()),
        help="Apply cross-encoder re-ranking for better accuracy"
    )

    # Number of results
    st.session_state.search_k = st.sidebar.slider(
        "Results to retrieve",
        min_value=1,
        max_value=10,
        value=getattr(st.session_state, 'search_k', config.get_search_k()),
        help="Number of document chunks to retrieve"
    )


def render_configuration_display(config: Any):
    """Render the configuration display expander in the sidebar.

    Args:
        config: The application config object.
    """
    with st.sidebar.expander("Configuration", expanded=False):
        st.markdown(f"**Embedding**: {config.get_embedding_model()}")
        st.markdown(f"**Chat Model**: {config.get_chat_model()}")
        st.markdown(f"**Chunk Size**: {config.get_chunk_size()}")

        # Show re-ranker with provider details
        reranker_provider = config.get_reranker_provider()
        if reranker_provider == "auto":
            st.markdown("**Re-ranker**: auto")
            st.caption("Priority: jina (local) → cohere (cloud)")
        else:
            st.markdown(f"**Re-ranker**: {reranker_provider}")
