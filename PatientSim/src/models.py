import os
import time
import json
import datetime
from openai import AzureOpenAI, OpenAI

try:
    from google import genai
    from google.genai import types
    from google.genai.types import HttpOptions
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

try:
    from dotenv import load_dotenv
except ImportError:
    pass

# load_dotenv(override=True)
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
PORT = os.environ.get("VLLM_PORT", "")

if AZURE_OPENAI_KEY != "":
    azure_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-10-21",
    )
else:
    # Fall back to DefaultAzureCredential (token-based auth, no API key needed)
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        _credential = DefaultAzureCredential()
        _token_provider = get_bearer_token_provider(
            _credential, "https://cognitiveservices.azure.com/.default"
        )
        _endpoint = AZURE_ENDPOINT or AZURE_OPENAI_ENDPOINT
        if _endpoint:
            azure_client = AzureOpenAI(
                azure_endpoint=_endpoint,
                azure_ad_token_provider=_token_provider,
                api_version="2024-12-01-preview",
            )
    except ImportError:
        pass  # azure-identity not installed; azure_client stays undefined

if GENAI_API_KEY != "" and _HAS_GENAI:
    gen_client = genai.Client(vertexai=True, api_key=GENAI_API_KEY, http_options=HttpOptions(api_version="v1"))

time_gap = {"gpt-4": 3, "gpt-4.1": 3}


def get_answer(response):
    if hasattr(response, "choices"):
        answer = response.choices[0].message.content
    elif hasattr(response, "text"):
        answer = response.text.strip()
    else:
        raise NotImplementedError(f"Fail to extract answer: {answer}")
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].replace("<think>", "").replace("\n", "").strip()
    return answer


def get_token_log(response):
    token_usage = {}
    if hasattr(response, "usage"):
        token_usage["prompt_tokens"] = response.usage.prompt_tokens
        token_usage["completion_tokens"] = response.usage.completion_tokens
        token_usage["total_tokens"] = response.usage.total_tokens
        if hasattr(response.usage, "completion_tokens_details"):  # for gpt-5 series
            if hasattr(response.usage.completion_tokens_details, "reasoning_tokens"):
                token_usage["extra_info"] = {"reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens}
    elif hasattr(response, "usage_metadata"):
        token_usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
        token_usage["completion_tokens"] = response.usage_metadata.candidates_token_count
        token_usage["total_tokens"] = response.usage_metadata.total_token_count
    else:
        raise NotImplementedError(f"Fail to extract usage data: {response}")
    return token_usage
    

def gpt_azure_response(message: list, model="gpt-4o", temperature=0, seed=42, **kwargs):
    wait = time_gap.get(model, 3)
    while True:
        time.sleep(wait)
        try:
            return azure_client.chat.completions.create(model=model, messages=message, temperature=temperature, seed=seed, **kwargs)
        except Exception as e:
            err = str(e).lower()
            if "content_filter" in err or "content management policy" in err:
                raise RuntimeError(f"Content filter blocked request: {e}")
            print(f"    [Azure unavailable: {e} -- waiting {wait}s...]")
            time.sleep(wait)


def gemini_response(message: list, model="gemini-2.0-flash", temperature=0, seed=42, **kwargs):
    time.sleep(time_gap.get(model, 3))
    system_prompt = message[0]["content"] if message[0]["role"] == "system" else None
    if system_prompt:
        contents = message[1:]
    else:
        contents = message

    try:
        contents = [{"role": item["role"], "parts": [{"text": item["content"]}]} for item in contents]
    except:
        raise NotImplementedError

    try:
        if model == "gemini-2.5-flash":
            return gen_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    seed=seed,
                    thinking_config=types.ThinkingConfig(thinking_budget=kwargs.get("thinking_budget", 0))
                ),
            )
        else:
            return gen_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    seed=seed,
                ),
            )

    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "length" in error_msg or 'maximum context length' in error_msg:
            if isinstance(message, list) and len(message) > 2:
                message = [message[0]] + message[2:]
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return gemini_response(message, model, temperature, seed, **kwargs)


def vllm_model_setup(model):
    if model == "vllm-llama3-70b-instruct":
        model = "meta-llama/Llama-3-70B-Instruct"
    elif model == "vllm-llama3-8b-instruct":
        model = "meta-llama/Llama-3-8B-Instruct"
    elif model == "vllm-llama3.1-8b-instruct":
        model = "meta-llama/Llama-3.1-8B-Instruct"
    elif model == "vllm-llama3.1-70b-instruct":
        model = "meta-llama/Llama-3.1-70B-Instruct"
    elif model == "vllm-llama3.3-70b-instruct":
        model = "meta-llama/Llama-3.3-70B-Instruct"
    elif model == "vllm-qwen2.5-72b-instruct":
        model = "Qwen/Qwen2.5-72B-Instruct"
    elif model == "vllm-qwen2.5-7b-instruct":
        model = "Qwen/Qwen2.5-7B-Instruct"
    elif model == "vllm-deepseek-llama-70b":
        model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    else:
        raise ValueError(f"Invalid model: {model}")
    return model


def vllm_response(message: list, model=None, temperature=0, seed=42, **kwargs):
    VLLM_API_KEY = "EMPTY"
    VLLM_API_BASE = f"http://localhost:{PORT}/v1"
    vllm_client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)

    assert model in [
        "meta-llama/Llama-3-70B-Instruct",
        "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ]
    time.sleep(time_gap.get(model, 3))

    try:
        return vllm_client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            seed=seed,
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "context" in error_msg or "length" in error_msg or 'maximum context length' in error_msg:
            if isinstance(message, list) and len(message) > 2:
                message = [message[0]] + message[2:]
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return vllm_response(message, model, temperature, seed)


def get_response_method(model):
    response_methods = {
        "gpt_azure": gpt_azure_response,
        "vllm": vllm_response,
        "genai": gemini_response,
    }
    return response_methods.get(model.split("-")[0], lambda _: NotImplementedError())