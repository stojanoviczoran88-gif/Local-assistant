import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional
import re

# Page configuration
st.set_page_config(
    page_title="CodeGuru - Local Code Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better code formatting and dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        background-color: #1e2329;
    }
    
    .user-message {
        border-left-color: #2196F3;
        background-color: #262730;
    }
    
    .assistant-message {
        border-left-color: #4CAF50;
        background-color: #1e2329;
    }
    
    .code-block {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    
    .status-online {
        background-color: #4CAF50;
    }
    
    .status-offline {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except:
            return []
    
    def generate_response(self, model: str, prompt: str, system_prompt: str = "", temperature: float = 0.1) -> str:
        """Generate response from Ollama model"""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 2048
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be taking too long to respond."
        except Exception as e:
            return f"Error: {str(e)}"

def get_system_prompt(model_type: str) -> str:
    """Get system prompt based on model type"""
    if "codellama" in model_type.lower() or "code" in model_type.lower():
        return """You are CodeGuru, an expert programming assistant. You specialize in:

- Writing clean, efficient, and well-documented code
- Debugging and troubleshooting code issues
- Explaining complex programming concepts clearly
- Providing best practices and optimization suggestions
- Supporting multiple programming languages and frameworks

Always provide:
1. Clear, concise explanations
2. Working code examples when relevant
3. Best practices and potential improvements
4. Error handling considerations

Format your responses with proper code blocks and clear structure."""
    
    else:  # Default for general models like Llama 3
        return """You are CodeGuru, a helpful programming assistant. You help with coding questions, debugging, explanations, and best practices. Provide clear, accurate, and practical advice for programming tasks."""

def format_code_response(text: str) -> str:
    """Format response text with proper code block highlighting"""
    # Replace code blocks with streamlit markdown
    code_pattern = r'```(\w+)?\n(.*?)\n```'
    
    def replace_code_block(match):
        language = match.group(1) or ''
        code = match.group(2)
        return f'\n```{language}\n{code}\n```\n'
    
    formatted = re.sub(code_pattern, replace_code_block, text, flags=re.DOTALL)
    return formatted

def main():
    # Initialize Ollama client
    ollama = OllamaClient()
    
    # Sidebar configuration
    st.sidebar.title("🤖 CodeGuru Settings")
    
    # Connection status
    is_connected = ollama.check_connection()
    status_color = "status-online" if is_connected else "status-offline"
    status_text = "Online" if is_connected else "Offline"
    
    st.sidebar.markdown(
        f'<div><span class="status-indicator {status_color}"></span>Ollama: {status_text}</div>',
        unsafe_allow_html=True
    )
    
    # Model selection
    if is_connected:
        available_models = ollama.get_available_models()
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Select Model",
                available_models,
                index=0
            )
        else:
            st.sidebar.error("No models found. Please install a model first.")
            st.sidebar.code("ollama pull codellama:7b")
            return
    else:
        st.sidebar.error("Ollama is not running. Please start Ollama first.")
        st.sidebar.code("ollama serve")
        return
    
    # Temperature setting
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower values make responses more deterministic"
    )
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Model info
    with st.sidebar.expander("ℹ️ Model Information"):
        st.write(f"**Current Model:** {selected_model}")
        if "codellama" in selected_model.lower():
            st.write("**Type:** Code-specialized model")
            st.write("**Best for:** Code generation, debugging, explanations")
        else:
            st.write("**Type:** General-purpose model")
            st.write("**Best for:** General coding assistance")
    
    # Main interface
    st.title("💻 CodeGuru - Local Code Assistant")
    st.markdown("*Your offline programming companion powered by Ollama*")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(format_code_response(message["content"]))
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about code..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("CodeGuru is thinking..."):
                system_prompt = get_system_prompt(selected_model)
                
                # Build conversation context
                conversation_context = ""
                for msg in st.session_state.messages[-5:]:  # Last 5 messages for context
                    if msg["role"] == "user":
                        conversation_context += f"User: {msg['content']}\n"
                    else:
                        conversation_context += f"Assistant: {msg['content']}\n"
                
                conversation_context += f"User: {prompt}\nAssistant: "
                
                response = ollama.generate_response(
                    model=selected_model,
                    prompt=conversation_context,
                    system_prompt=system_prompt,
                    temperature=temperature
                )
                
                formatted_response = format_code_response(response)
                st.markdown(formatted_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer with usage instructions
    with st.expander("📖 Usage Tips"):
        st.markdown("""
        **Getting Started:**
        - Make sure Ollama is running (`ollama serve`)
        - Install models: `ollama pull codellama:7b` or `ollama pull llama3`
        
        **Best Practices:**
        - Be specific about the programming language and context
        - Ask for code reviews, explanations, or debugging help
        - Use lower temperature (0.1-0.3) for more consistent code generation
        
        **Example Prompts:**
        - "Write a Python function to sort a list of dictionaries by a specific key"
        - "Debug this JavaScript code: [paste your code]"
        - "Explain how async/await works in Python"
        - "Optimize this SQL query for better performance"
        """)

if __name__ == "__main__":
    main()