# Simple AI Assistant using LangChain and Hugging Face (No API Key)

This project demonstrates how to build a **local AI chatbot** using **LangChain**, **Transformers**, and an open-source model from Hugging Face. It works in Google Colab with GPU acceleration and does **not require any OpenAI API key**.

---

## Features

- ✅ Uses open-source model `google/flan-t5-xl`
- ✅ No API key required (completely free)
- ✅ Uses LangChain for prompt templating
- ✅ Runs on GPU in Colab using `transformers.pipeline`
- ✅ Easy to modify for Q&A, summaries, explanations, etc.

---

## Sample Output

**Q:** Who is the founder of Google?  
**A:** Google was founded by Larry Page and Sergey Brin.

---

## Setup Instructions

### 1. Install Required Libraries

```bash
pip install transformers accelerate langchain
```

### 2. Load the Model

```python
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

model_id = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

pipe = pipeline("text2text-generation", model="google/flan-t5-xl", device=0)
llm = HuggingFacePipeline(pipeline=pipe)
```

### 3. Define the Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | llm
```

### 4. Ask a Question

```python
response = chain.invoke({"input": "What is the capital of Russia?"})
print("Answer:", response)
```

### 5. Conclusion

This AI Assistant demonstrates how we can use open-source language models locally without relying on paid APIs. While it can handle basic factual questions and simple explanations, it has some limitations:

❌ It does not provide real-time or updated information (e.g., current events or latest data).

❌ It may repeat or generalize responses due to the nature of the model.

❌ It struggles with complex tasks that require reasoning or dynamic knowledge.

For complete and accurate results on up-to-date topics, real-time access to APIs like OpenAI, Tavily, or web search plugins is recommended.
