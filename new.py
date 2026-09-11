import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Langchain: Summarize Text from YouTube or Website", page_icon="🦜")
st.title("🦜 Langchain: Summarize Text from YouTube or Website")
st.subheader("Summerize URL")

with st.sidebar:
    groq_api_key=st.text_input("Please enter Groq API key", value="", type='password')

llm=ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_api_key)

generic_prompt="""
        Please provide a summary of following content in 300 words:
        Content:{text}
"""

prompt=PromptTemplate(input_variables=['text'],template=generic_prompt)

url=st.text_input("URL",label_visibility="collapsed")

if st.button("Summarize the content from Youtube or Website"):
    if not groq_api_key.strip() or not url.strip():
        st.warning("Please provide information to get started")
    elif not validators.url(url):
        st.warning("Please enter valid url")
    else:     
            with st.spinner("Waiting..."):
                if 'youtube.com' in url:
                    loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[url], ssl_verify=True,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})

                docs=loader.load()
                chain=load_summarize_chain(
                    llm=llm,prompt=prompt, chain_type='stuff'
                    )
                output=chain.run(docs)
                st.success(output)
        
