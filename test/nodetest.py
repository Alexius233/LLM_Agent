from llama_index.llms import ChatMessage, HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import SimpleDirectoryReader
from llama_index.text_splitter import SentenceSplitter
from llama_index import VectorStoreIndex
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.retrievers import VectorIndexRetriever
import pprint

reader = SimpleDirectoryReader(input_files=["/yuanzichen/LlamaChat/visual.pdf"])
documents = reader.load_data()

node_parser = SentenceSplitter(chunk_size=512,chunk_overlap=128) 
# 指定chunk的大小,指定overlap,也可以不指定，自动计算
# 这个应该是模糊值,主要的分割策略是保持段落和句子的完整性

# use transforms directly
nodes = node_parser(documents)

index = VectorStoreIndex(nodes)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)
# 或者这样,使用Index转换成默认的retriever
# retriever = index.as_retriever()

query_text = "Introduction of the paper"
# 得到的就是Nodes
Nodes = retriever.retrieve(query_text)

print(len(Nodes))
print(Nodes[0].text)