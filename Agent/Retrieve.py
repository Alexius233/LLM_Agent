from llama_index.llms import ChatMessage
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from enum import Enum
from llama_index.vector_stores.types import VectorStore


class ToolTypes(str, Enum):
    """Tool Types. 用来call RetrievalTools"""

    LOCAL = "local"
    ONLINE = "online"
    WEB = "web"
    STORAGE = "storage"


class RetrievalTools:
    def __inint__(
            self,
            model_path:str,
            embed_path:str = None,
    ) -> None:

        self._model_path = model_path
        self._embed_path = (embed_path if embed_path is not None 
                            else "path/to/all-MiniLM-L6-v2")
        
    
    def get_tokenizer_model(self):
        model = AutoModelForCausalLM.from_pretrained(self._model_path)
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        return model,tokenizer

    def get_global_service_context(self, model, tokenizer) -> None:

        llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    model=model,
                    tokenizer=tokenizer)
        
        embeddings=LangchainEmbedding(
        HuggingFaceEmbeddings(model_name=self._embed_path)
        )
        # Create new service context instance
        service_context = ServiceContext.from_defaults(
            chunk_size=1024,
            llm=llm,
            embed_model=embeddings
        )
        # And set the service context
        set_global_service_context(service_context)
    
    def process_local(self, index_path) -> VectorStore:
        
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)

        return index

    
    def process_online(self, file_path) -> VectorStore:
        
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)

        return index
    
    def process_web(self) -> VectorStore:
        None  # 爬取web端，先空着
    
    def storage(self, index, storage_path) -> None:
        
        index.storage_context.persist(persist_dir=storage_path)


    def retrieve(self, tool_type, *args, **kwargs):
        model, tokenizer = self.get_tokenizer_model()
        self.get_global_service_context(model, tokenizer)


        if tool_type == ToolTypes.LOCAL:
            return self.process_local(*args, **kwargs)
        elif tool_type == ToolTypes.ONLINE:
            return self.process_online(*args, **kwargs)
        elif tool_type == ToolTypes.WEB:
            return self.process_web(*args, **kwargs)
        elif tool_type == ToolTypes.STORAGE:
            return self.storage(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported tool type: {tool_type}")
