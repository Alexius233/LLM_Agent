import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes 
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents 
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from pathlib import Path


# Define variable to hold llama2 weights naming 
name = "/yuanzichen/LlamaChat/Llama2ChatHF"
# Set auth token variable from hugging face 
auth_token = "YOUR HUGGING FACE AUTH TOKEN HERE"

@st.cache_resource
def get_tokenizer_model():
    """
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/')

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                            , torch_dtype=torch.float16, 
                            rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)
    """
    tokenizer = AutoTokenizer.from_pretrained("/yuanzichen/LlamaChat/Llama2ChatHF")
    model = AutoModelForCausalLM.from_pretrained("/yuanzichen/LlamaChat/Llama2ChatHF")

    return tokenizer, model
tokenizer, model = get_tokenizer_model()

# Create a system prompt 
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of 
the company.<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper 
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)
#llm.chat()

# Create and dl embeddings instance  
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="/yuanzichen/LlamaChat/sentence-transformers/all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)


# Create PDF Loader
reader = SimpleDirectoryReader(input_files=["/yuanzichen/LlamaChat/visual.pdf"])
# Load documents 
documents = reader.load_data()

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)
# Setup index query engine using LLM 
query_engine = index.as_query_engine()
print("11111111111111111111")
"""
# Create centered main title 
st.title('🦙 Llama Banker')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    response = query_engine.query(prompt)
    # ...and write it out to the screen
    st.write(response)

    # Display raw response object
    with st.expander('Response Object'):
        st.write(response)
    # Display source text
    with st.expander('Source Text'):
        st.write(response.get_formatted_sources())
"""

print('🦙 Llama Banker')
print('Enter "quit" to exit.')

while True:  # 开始一个无限循环
    prompt = input('Input your prompt here: ')

    if prompt.lower() == 'quit':  # 如果用户输入 "quit"，则退出循环
        break

    if prompt:  # 如果有输入
        # 处理输入并生成响应的代码
        response = query_engine.query(prompt)
        print(response)
        # 可以添加其它输出，如响应对象或源文本

# 循环结束后，可以添加一些清理或结束语
print('Goodbye!')