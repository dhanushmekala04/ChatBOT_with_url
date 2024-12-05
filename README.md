
### Overview

This project creates a Streamlit app that allows users to chat with websites using AI. It uses LangChain to retrieve information from web pages and generate responses to user queries.

### Process Overview

1. The app loads content from a specified website URL.
2. It processes the content into smaller chunks and creates embeddings.
3. When a user asks a question, the app retrieves relevant information from the processed content.
4. An AI model (Groq LLM) generates a response based on the retrieved context.

### Prerequisites

- Python 3.x
- Streamlit
- LangChain and its dependencies
- Groq API key

### Set Up a Virtual Environment

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install required packages:
   ```
   pip install streamlit langchain requests beautifulsoup4 faiss-cpu langchain-huggingface langchain-groq
   ```

### Install Requirements

After setting up the virtual environment, install the project requirements:

```
pip install -r requirements.txt
```

### Run the Streamlit App

To run the app, use the following command:

```
streamlit run main.py
```

Replace `main.py` with the actual filename of your Streamlit script.

### Project Structure

```
project_root/
│
├── main.py
├── requirements.txt
└── README.md
```

- `main.py`: The main Streamlit application file.
- `requirements.txt`: List of Python packages required to run the project.
- `README.md`: This document containing setup instructions and project overview.

### Additional Notes

- Make sure to replace the placeholder Groq API key (``) with your actual Groq API key.
- Adjust the `DB_DIR` constant as needed to specify where you want to store the vector database files.
- Ensure that the website URL provided is accessible and contains relevant information for the chat functionality.

### Video Demonstration
[![Watch the video](https://raw.githubusercontent.com/username/repository/main/path/to/video.mp4)
