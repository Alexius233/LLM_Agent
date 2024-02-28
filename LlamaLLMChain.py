from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain


# LLMChain 用来自动生成自定义的角色对话

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Customer: {question}
Assistant:""")

model_path = "/yuanzichen/LlamaChat/Llama2ChatHF"

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
chain = LLMChain(llm=llm, prompt=prompt)
res = chain.run(question="hello")

print(res)