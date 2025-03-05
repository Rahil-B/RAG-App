from langchain_community.document_loaders import WebBaseLoader

def fetch_data_from_urls(urls):
    docs = []
    for url in urls:
        if isinstance(url, str) and url.strip():
            try:
                loader = WebBaseLoader(url)
                loaded_docs = loader.load()
                print(f"Loaded {len(loaded_docs[0].page_content)} char from {url}")
                docs.extend(loaded_docs)
            except Exception as e:
                print(f"Cant load URL {url}: {e}")
    #print(docs)
    if not docs:
        return []
    # docs = [WebBaseLoader(url).load() for url in urls if type(url)!=str and url.strip() != ""]
    docs_list = [item.page_content  for item in docs]
    # clear log file and then loade data to a log file
    with open("web_content_log.txt", "w") as f:
        for item in docs_list:
            f.write(f"Content: {item}\n")
    
    return docs_list