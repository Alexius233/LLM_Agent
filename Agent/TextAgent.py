import json
from typing import Sequence, List

from llama_index.llms import ChatMessage
from llama_index.tools import BaseTool, FunctionTool
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever

import nest_asyncio

# llamaindex的openai agent实现

model_path = "/yuanzichen/LlamaChat/Llama2ChatHF"

nest_asyncio.apply()

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

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

class AIAgent:
    def __init__(
        self,
        model,
        system_prompt,
        query_wrapper_prompt,
        tools: Sequence[BaseTool] = [],
        chat_history: List[ChatMessage] = [],
    ) -> None:
        
        self._model = model
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history
        self._system_prompt = system_prompt
        self._query_wrapper_prompt = query_wrapper_prompt

        self._llm = HuggingFaceLLM(context_window=4096,
                        max_new_tokens=256,
                        system_prompt=self._system_prompt,
                        query_wrapper_prompt=self._query_wrapper_prompt,
                        model=self._model,
                        tokenizer=tokenizer)


    def get_history(self) -> List:
        return self._chat_history

    def update_history(self, message) -> List:
        return self._chat_history.append(message)
    
    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self.get_history
        chat_history.append(ChatMessage(role="user", content=message))
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]

        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        
        self.update_history(ai_message)
    
        tool_calls = ai_message.additional_kwargs.get("tool_calls", None)
        # parallel function calling is now supported
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, tool_call: dict) -> ChatMessage:
        id_ = tool_call.id
        function_call = tool_call.function
        tool = self._tools[function_call.name]
        output = tool(**json.loads(function_call.arguments))
        print(f"> Calling tool: {function_call.name}")
        return ChatMessage(
            name=function_call.name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call.name,
            },
        )
    

agent = AIAgent(
    model=model,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tools=[multiply_tool, add_tool])

response = agent.chat("I am boring now, tell a story to me")

print(str(response))

#print("######################################################")

#response1 = agent.chat("Give me the summary of this story")

#print(str(response1))
