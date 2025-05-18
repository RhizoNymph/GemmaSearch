# GemmaSearch

This is a basic implementation of function calling intended for use with Gemma3-27b-it with:

- DuckDuckGo search
- Arxiv search
- Playwright based web page fetch
- PDF text parsing on fetch

To run it, use:
```uv sync && uv run playwright install```

followed by:
```uv run gemma3.py```

Right now it assumes you have an openai server running Gemma3 on your localhost (I'll fix this soon sorry).
I run mine using llama-server with: 
```./build/bin/llama-server -m ./models/gemma-3-27b-it-q4_0.gguf -c 8192 -fa --host 0.0.0.0 -ngl 100 -cb -mg 1```
(assumes model file is in models folder)

To build llama-server use:
```
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release -t llama-server
```

Todo:
- Add environment variables for server
- Add web search options that aren't DuckDuckGo (has rate limit issues)
- Add RAG setup and context management for long documents/long running sessions
- Add support for other models