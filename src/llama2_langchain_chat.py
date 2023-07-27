import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from src.llama2_7b import get_prompt

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=True,)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                            #  load_in_8bit=True,
                                            #  load_in_4bit=True
                                             )

pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})

instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
system_prompt = "You are a helpful assistant, you always only answer for the assistant then you stop. read the chat history to get context"

template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

llm_chain.predict(user_input="Hi, my name is Sam")

llm_chain.predict(user_input="Can you tell me about yourself.")

llm_chain.predict(user_input="Today is Friday. What number day of the week is that?")

llm_chain.predict(user_input="what is the day today?")

llm_chain.predict(user_input="What is my name?")

llm_chain.predict(user_input="Can you tell me about the olympics")

llm_chain.predict(user_input="What have we talked about in this Chat?")