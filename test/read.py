from llama_index.llms import ChatMessage, HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.retrievers import VectorIndexRetriever
from llama_index import StorageContext, load_index_from_storage
import pprint

from llama_index.text_splitter import SentenceSplitter

# 看起来from document需要llm, 可以试试手动分割成nodes 或者 disable llm
# index = VectorStoreIndex(nodes)

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

llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

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

storage_context = StorageContext.from_defaults(persist_dir="/yuanzichen/LlamaChat/storage")

index = load_index_from_storage(storage_context)