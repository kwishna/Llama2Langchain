import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from src.llama2_7b import parse_text, get_prompt

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=True, )

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                             #  load_in_8bit=True,
                                             #  load_in_4bit=True
                                             )

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

system_prompt = "You are an advanced assistant that excels at translation. "
instruction = "Convert the following text from English to French:\n\n {text}"
template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

text = "how are you today?"
output = llm_chain.run(text)

parse_text(output)

"""### Summarization"""

instruction = "Summarize the following article for me {text}"
system_prompt = "You are an expert and summarization and expressing key ideas succintly"

template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


def count_words(input_string):
    words = input_string.split(" ")
    return len(words)


text = '''Twitter (now X) CEO Linda Yaccarino claims usage at ‘all time high’ in memo to staff
Twitter’s (now X’s) newly established CEO Linda Yaccarino touts the company’s success and X’s future plans in a company-wide memo obtained by CNBC. The exec once again claims, without sharing any specific metrics, that the service’s usage is at an “all time high,” and hints at what’s to come in terms of new product experiences for the newly rebranded platform.

The service formerly known as Twitter has been working to become more than just a social network and more of an “everything app,” as owner Elon Musk dubbed it.

As the Telsa and Space X exec explained in October 2022, telegraphing Twitter’s eventual rebranding, buying Twitter was meant to be “an accelerant to creating X, the everything app.”


His grand plan has been to create an app that allows creators to monetize their content, then later moves into payments services and even banking, Musk remarked during a Twitter Spaces livestream with advertisers in November. At the time, he even mentioned the possibility of establishing money market accounts on Twitter that would pay a high-interest rate to attract consumers to X.

Those possible product concepts were again referenced in Yaccarino’s new missive, when she writes, “Our usage is at an all time high and we’ll continue to delight our entire community with new experiences in audio, video, messaging, payments, banking – creating a global marketplace for ideas, goods, services, and opportunities.”

Twitter, now X, has already implemented some of Musk’s ideas around videos and creator monetization. In May, the company began allowing subscribers to upload two-hour videos to its service, which advertiser Apple then leveraged when it released the entire first episode of its hit Apple TV+ show “Silo” on the platform. Fired Fox News host Tucker Carlson had been posting lengthy videos to Twitter as well, until ordered to stop by the network.

In addition, earlier this month, Twitter began sharing ad revenue with verified creators.

However, all is not well at Twitter X, whose traffic — at least by third-party measurements — has been dropping. Data from web analytics firm Similarweb indicated Twitter’s web traffic declined 5% for the first two days its latest rival, Instagram Threads, became generally available, compared with the week prior. Plus, Similarweb said Twitter’s web traffic was down 11% compared with the same days in 2022. Additionally, Cloudflare CEO Matthew Prince earlier this month tweeted a graph of traffic to the Twitter.com domain that showed “Twitter traffic tanking,” he said.


Yaccarino subtly pushed back at those reports at the time, claiming that Twitter had its largest usage day since February in early July. She did not share any specific metrics or data. At the same time, however, the company was quietly blocking links to Threads.net in Twitter searches, suggesting it was concerned about the new competition.

Today, Yaccarino repeats her vague claims around X’s high usage in her company-wide memo even as analysts at Forrester are predicting X will either shut down or be acquired within the next 12 months and numerous critics concur that the X rebrand is destined to fail.

Yaccarino’s memo, otherwise, was mostly a lot of cheerleading, applauding X’s team for their work and touting X’s ability to “impress the world all over again,” as Twitter once did.

The full memo, courtesy of CBNC, is below:

Hi team,

What a momentous weekend. As I said yesterday, it’s extremely rare, whether it’s in life or in business, that you have the opportunity to make another big impression. That’s what we’re experiencing together, in real time. Take a moment to put it all into perspective.

17 years ago, Twitter made a lasting imprint on the world. The platform changed the speed at which people accessed information. It created a new dynamic for how people communicated, debated, and responded to things happening in the world. Twitter introduced a new way for people, public figures, and brands to build long lasting relationships. In one way or another, everyone here is a driving force in that change. But equally all our users and partners constantly challenged us to dream bigger, to innovate faster, and to fulfill our great potential.

With X we will go even further to transform the global town square — and impress the world all over again.

Our company uniquely has the drive to make this possible. Many companies say they want to move fast — but we enjoy moving at the speed of light, and when we do, that’s X. At our core, we have an inventor mindset — constantly learning, testing out new approaches, changing to get it right and ultimately succeeding.

With X, we serve our entire community of users and customers by working tirelessly to preserve free expression and choice, create limitless interactivity, and create a marketplace that enables the economic success of all its participants.

The best news is we’re well underway. Everyone should be proud of the pace of innovation over the last nine months — from long form content, to creator monetization, and tremendous advancements in brand safety protections. Our usage is at an all time high and we’ll continue to delight our entire community with new experiences in audio, video, messaging, payments, banking – creating a global marketplace for ideas, goods, services, and opportunities.

Please don’t take this moment for granted. You’re writing history, and there’s no limit to our transformation. And everyone, is invited to build X with us.

Elon and I will be working across every team and partner to bring X to the world. That includes keeping our entire community up to date, ensuring that we all have the information we need to move forward.

Now, let’s go make that next big impression on the world, together.

Linda'''

count_words(text)

output = llm_chain.run(text)
print(count_words(output))
parse_text(output)
