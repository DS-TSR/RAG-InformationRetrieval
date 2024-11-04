# RAG-InformationRetrieval
RAG-based system to streamline the extraction and analysis of key information From Financial Reports.
# MLS: Finsights Grey - RAG for Effective Information Retrieval
[Github]
[VSCode]
[AzureOpenAILab]

# Business Use Case

# Problem Statement:

Finsights Grey Inc. is an innovative financial technology firm that specializes in providing advanced analytics and insights for investment management and financial planning. The company handles an extensive collection of 10-K reports from various industry players, which contain detailed information about financial performance, risk factors, market trends, and strategic initiatives. Despite the richness of these documents, Finsights Grey's financial analysts struggle with extracting actionable insights efficiently in a short span due to the manual and labor-intensive nature of the analysis. Going through the document to find the exact information needed at the moment takes too long. This bottleneck hampers the company's ability to deliver timely and accurate recommendations to its clients. To overcome these challenges, Finsights Grey Inc. aims to implement a Retrieval-Augmented Generation (RAG) model to automate the extraction, summarization, and analysis of information from the 10-K reports, thereby enhancing the accuracy and speed of their investment insights.

# Objective:

As a Gen AI Data Scientist hired by Finsights Grey Inc., the objective is to develop an advanced RAG-based system to streamline the extraction and analysis of key information from 10-K reports.

The project will involve testing the RAG system on a current business problem. The Financial analysts are asked to research major cloud and AI platforms such as Amazon AWS, Google Cloud, Microsoft Azure, Meta AI, and IBM Watson to determine the most effective platform for this application. The primary goals include improving the efficiency of data extraction. Once the project is deployed, the system will be tested by a financial analyst with the following questions. Accurate text retrieval for these questions will imply the project's success.

# Questions:

1. Has the company made any significant acquisitions in the AI space, and how are these acquisitions being integrated into the company's strategy?

2. How much capital has been allocated towards AI research and development?

3. What initiatives has the company implemented to address ethical concerns surrounding AI, such as fairness, accountability, and privacy?

4. How does the company plan to differentiate itself in the AI space relative to competitors?

Each Question must be asked for each of the five companies.

By successfully developing this project, we aim to:

Improve the productivity of financial analysts by providing a competent tool.

Provide timely insights to improve client recommendations.

Strengthen FinTech Insights Inc.’s competitive edge by delivering more reliable and faster insights to clients.

# Clone Repositorory

[git clone https://github.com/username/Sentiment_Analysis.git] at the folder location.

# Installations

!pip install -q openai==1.23.2 \
                tiktoken==0.6.0 \
                pypdf==4.0.1 \
                langchain==0.1.1 \
                langchain-community==0.0.13 \
                chromadb==0.4.22 \
                sentence-transformers==2.3.1
# Imports

Pip install requirements.txt

# Impementing RAG

# Prepare Data

Let's start by loading the dataset.

# Chunking

Let's split the contents of the pdf into chunks of size 512 (as this is the max size allowed by the embedding model we have choosen. Leet's also have some overlap between the chunks. 16 token should give us 2 sentences of overlap.

from the RAG file we canObserve the structure of the chunk. Notice the metadata section and how it has a source and page number.

# Database Creation

#Create a Colelction Name

#Initiate the embedding momdel 'thenlper/gte-large'


The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(

 # Create the vector Database

 Once we have created the vectorstore, we do not need the GPU. So, we can switch to a CPU instance on Google Colab. But when we switch, we will lose the vectorDB that we have created in this session. To persist the DB across sessions lets persist it and then save it/download it so that we can reuse it in a different session.

 # Persist the DB

 Let us save our vectorDB in our Google Drive so that we can retrieve it later whenever we want. Provide Google Colab access to your Gdrive when prompted.

 #Mount the Google Drive

 #Copy the persisted database to your drive

 # Test your RAG Application

 Prevent over using GPU but if you want you can shift to CPU now but i continued to Use GPU.

 # Load Vector DB from Google Drive

Let's test our database with a sample question.
Say, the financial markets are responding positively to AI, then we would like to know which companies have aggresively integrated AI in their business units.

Provide User Question.
#Perform similarity search on the user_question

#You must add an extra parameter to the similarity search  function so that you can filter the response based on the 'source'  in the metadata of the doc

#The filter can be added as a parameter to the similarity search function.

#This will allow you to retrieve chunks from a particular document.

#Use the same format to filter your response based on the company.

#Print the retrieved docs, their source and the page number.

#(page number can be accessed using doc.metadata['page'] )

# RAG Q&A

# Prompt Design

#Create a system message for the LLM

#Create a message template

# Composing the response

#Create a variable company to store the source of the context so that you can filter the similarity search

company = "dataset/google-10-k-2023.pdf" # We shall change this programmatically later when we test on multiple queries for each of the company

#Create context for query by joining page_content and page number of the retrieved docs

#Craft the messages to pass to chat.completions.create

#Get a response from the LLM

#Handle errors using try-except

#Answer

# Evaluation

It is important to note that all the answers may not be found in the documents provided to the RAG system. The RAG system can be augmented with further information like new and legitimate sources from internet. In any case, when the information is not available to the RAG system, we want it to state the same instead of 'halucinating' information which can lead to very bad consequences. Hence, we need to evaluate the RAG not only on how well it answers a question (relevance) but also on how much of the answer comes from the context provided (Groundedness).

#Create a prompt for the rater LLM to check the groundedness of the response

#Create a prompt for the rater LLM to check the relevance of the response

#Create user message template such that question, answer and context can be provided through it.

Let's manually check our RAG and evaluate its responses on a sample question.

To evaluate the company's investment strategy, it is essential to identify whether it aligns with current market trends or is directed towards sectors experiencing significant downturns. This analysis will enable financial analysts to make informed decisions.

For instance, if the cryptocurrency sector is facing technological challenges and consequently has a negative market sentiment, companies with substantial investments in this area are likely to experience adverse effects as well. Understanding these dynamics will provide valuable insights into the potential risks and opportunities associated with the company's investment choices.

#Create context

#Create the messages for chat.completion.create()

#Get a response from the LLM

#Handle errors using try-except

Retrieved chunk 1

We can see that this company is not majorly investing in crypto.

#Create messages for groundness

#Print the response of the rater LLM on groundednes

print(response.choices[0].message.content)

#Print the response of the rater LLM on relevance

# evaluation On Multiple Queries

Check the different groundedness scores and relevance scores obtained for our questions across different companies. As you can see when there is no relevant context provided, the model responds most of the time with "Sorry, this is out of my knowledge base" and when the context is provided, it responds with a decent answer.

For Example, we asked the question - "What initiatives has the company implemented to address ethical concerns surrounding AI, such as fairness, accountability, and privacy?
" w.r.t google and the model summarises the context provided to it and answer the question by talking about how google prioritises beneficial use, users safty and avoidance of harms above business consideration.

While the same question asked to msft, the model emphasises microsofts philosophy of getting AI to everyone but by keeping ethics, and fairness at the forefront.

And the the same question to aws return nothing showing the lack of strong ethical guideliness/principles assumed by amazon regarding AI. This could be an important insight.

Identifying such differences can enable financial analysts to make prudent decisions. For example, if the government introduces new regulations requiring companies developing AI solutions to adhere to a specific set of ethical standards, this could influence the stock performance of these companies based on their compliance with these ethical commitments.

Overall our model does a good job of providing relevant information while being grounded in the information provided.