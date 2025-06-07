from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
## user prompt

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(template= "generate a detailed report on {topic}", input_variables=["topics"])

prompt2 = PromptTemplate(template= "generate a 5 pointer summary from the following text\n {text}", input_variables=["text"])


parser = StrOutputParser()

# Step 1: Generate detailed report
report_chain = prompt1 | model | parser


# Step 2: Generate summary from report
summary_chain = prompt2 | model | parser

# Run chain step-by-step
input_data = {"topic": "poverty in India"}
report = report_chain.invoke(input_data)
print("📄 Detailed Report:\n", report)

summary = summary_chain.invoke({"text": report})
print("\n🔍 5-Point Summary:\n", summary)



# result = chain.invoke({"topic":"poverty in India"})

# print(result)

summary_chain.get_graph().print_ascii()