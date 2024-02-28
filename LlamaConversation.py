from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.llms import HuggingFacePipeline

model_path = "/yuanzichen/LlamaChat/Llama2ChatHF"

search = DuckDuckGoSearchAPIWrapper()

search_tool = Tool(name="Current Search",
                   func=search.run,
                   description="Useful when you need to answer questions about nouns, current events or the current state of the world."
                   )

#tools = [search_tool]
tools = []

memory = ConversationBufferMemory(memory_key="chat_history")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=256, 
    temperature=0.8
)

llm = HuggingFacePipeline(pipeline=pipe)

agent_chain = initialize_agent(tools,
                               llm,
                               agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                               memory=memory,
                               verbose=True)

agent_chain.run(input="What are you doing now?")



