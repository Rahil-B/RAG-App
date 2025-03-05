from langchain_community.document_loaders import WebBaseLoader

def fetch_data_from_urls(urls):
    loader = WebBaseLoader()
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item.page_content for sublist in docs for item in sublist]
    return docs_list