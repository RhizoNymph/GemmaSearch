import os, re, json, html
import openai
from typing import List, Dict, Optional, Tuple
from tools.parse import parse_tool_call_gemma, get_observation
from colorama import init, Fore, Style, Back
import utils.playwright_manager

# Initialize colorama
init()

client = openai.OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "sk-local-dummy"),
)

MODEL_NAME = "./models/gemma-3-27b-it-q4_0.gguf"

GEN_KWARGS = {
    "temperature": 0.95,
    "top_p": 0.7
}

def llm(messages: List[Dict[str, str]]) -> str:
    """Synchronous call to /v1/chat/completions."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        **GEN_KWARGS,
    )

    full_response = ""
    print(f"{Fore.CYAN}Assistant: {Style.RESET_ALL}", end="", flush=True)
    for chunk in resp:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    return full_response.strip()

def parse_function_call(response: str) -> Optional[Tuple[str, Dict]]:
    """Parse a function call from the model's response."""
    tool_call = parse_tool_call_gemma(response)
    if not tool_call:
        return None
        
    try:
        func_call = tool_call.strip()
        func_name = func_call.split("(")[0].strip()
        args_str = func_call[func_call.find("(")+1:func_call.rfind(")")]
        
        args = {}
        if args_str:
            for arg in args_str.split(","):
                if "=" in arg:
                    key, value = arg.split("=")
                    key = key.strip()
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    elif value.replace(".", "").isdigit():
                        value = float(value)
                    elif value.startswith('"') or value.startswith("'"):
                        value = value[1:-1]
                    args[key] = value
                    
        return func_name, args
    except Exception as e:
        print(f"Error parsing function call: {e}")
        return None

def run_agent_loop(initial_query: str, max_turns: int = 10) -> None:
    """Run the agent loop with function calling."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": initial_query}
    ]
    
    turns = 0
    while turns < max_turns:
        response = llm(messages)
        
        if "```tool_code" not in response:
            user_input = input(f"\n{Fore.GREEN}Your response (or press Enter to finish): {Style.RESET_ALL}").strip()
            if not user_input:
                print(f"\n{Fore.YELLOW}Ending conversation.{Style.RESET_ALL}")
                return
                
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": user_input})
            turns += 1
            continue
            
        messages.append({"role": "assistant", "content": response})
        
        func_call = parse_function_call(response)
        if not func_call:
            turns += 1
            continue
            
        func_name, args = func_call
        
        if func_name == "finish":
            print(f"\n{Fore.YELLOW}Agent finished the current response.{Style.RESET_ALL}")
            user_input = input(f"\n{Fore.GREEN}Would you like to continue? (y/n): {Style.RESET_ALL}").strip().lower()
            if user_input != 'y':
                print(f"\n{Fore.YELLOW}Ending conversation.{Style.RESET_ALL}")
                return
            messages.append({"role": "user", "content": "Please continue the conversation."})
            turns += 1
            continue

        try:
            observation = get_observation(func_name, args)

            if func_name in ["search_ddg", "search_arxiv"]:
                print(f"\n{Fore.BLUE}Search Results:{Style.RESET_ALL}")
                print(f"{Back.BLUE}{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}")
                print(observation)
                print(f"{Back.BLUE}{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}")
            elif func_name in ["click", "open"]:
                print(f"\n{Fore.MAGENTA}{func_name.capitalize()} result:{Style.RESET_ALL}")
                print(f"{Back.MAGENTA}{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}")
                print(observation)
                print(f"{Back.MAGENTA}{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}")
            messages.append({
                "role": "user",
                "content": f"```tool_output\n{observation}\n```"
            })
        except Exception as e:
            error_msg = f"Error executing {func_name}: {str(e)}"
            print(f"\n{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            messages.append({
                "role": "user",
                "content": f"```tool_output\n{error_msg}\n```"
            })
        
        turns += 1

sys_prompt = """
At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.
When parsing search results, you should use the click() function to click on the most relevant search result(s).
To finish the conversation, you must call the finish() function by wrapping it in a tool_code block like this:
```tool_code
finish()
```

The following Python methods are available:

```python
def search_ddg(query: str, k: int = 10) -> str:
    '''Search the web using DuckDuckGo for the most relevant information on the given query.'''
    Args:
        query (str): The search query.
        k (int): The number of results to return.
    Returns:
        str: A formatted string of search results.
    '''

def search_arxiv(query: str, k: int = 10) -> str:
    '''Search arXiv for academic papers matching the given query.'''
    Args:
        query (str): The search query.
        k (int): The number of results to return.
    Returns:
        str: A formatted string of arXiv paper results.
    '''

def click(rank: int) -> str:
    '''Click on a search result.'''
    Args:
        rank (int): The rank of the search result to click on.
    Returns:
        str: A formatted string of the clicked search result.
    '''

def open(url: str) -> str:
    '''Open a URL in playwright.'''
    Args:
        url (str): The URL to open.
    Returns:
        str: A formatted string of the opened URL.
    '''

def finish() -> None:
    '''Finish the conversation.'''
    Args:
        None
    Returns:
        None
    '''
```
"""

if __name__ == "__main__":
    initial_query = input("Enter your query: ")
    run_agent_loop(initial_query) 