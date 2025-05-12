import os, re, json, html
import openai
from typing import List, Dict, Optional, Tuple
from tools.parse import parse_tool_call_gemma, get_observation

import utils.playwright_manager

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
    for chunk in resp:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    print()
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
            print("\nAgent's response:")
            print(response)
            
            user_input = input("\nYour response (or press Enter to finish): ").strip()
            if not user_input:
                print("\nEnding conversation.")
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
            print("\nAgent finished the conversation.")
            return  

        try:
            observation = get_observation(func_name, args)

            if func_name == "search":
                print("\nSearch Results:")
                print("=" * 80)
                print(observation)
                print("=" * 80)
            if func_name in ["click", "open"]:
                print(f"\n{func_name.capitalize()} result:")
                print("=" * 80)
                print(observation)
                print("=" * 80)
            messages.append({
                "role": "user",
                "content": f"```tool_output\n{observation}\n```"
            })
        except Exception as e:
            error_msg = f"Error executing {func_name}: {str(e)}"
            print(f"\nError: {error_msg}")
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
def search(query: str, k: int = 10) -> str:
    '''Search the web for the most relevant information on the given query.'''
    Args:
        query (str): The search query.
        k (int): The number of results to return.
    Returns:
        str: A formatted string of search results.
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
    initial_query = "Give me a study plan for transitioning to robotics as a software engineer, based on search results"
    run_agent_loop(initial_query) 