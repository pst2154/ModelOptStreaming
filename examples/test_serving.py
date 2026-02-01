#!/usr/bin/env python3
"""Test script for vLLM NVFP4 serving."""

import time
from openai import OpenAI

# Configuration
SERVER_URL = "http://localhost:8000/v1"
MODEL_NAME = "kimi-k2.5-nvfp4"

# Initialize client
client = OpenAI(
    api_key="EMPTY",
    base_url=SERVER_URL,
    timeout=3600
)

# Test 1: Simple text completion
print("=== Test 1: Text Completion ===")
messages = [
    {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
    }
]

start = time.time()
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    max_tokens=512
)
elapsed = time.time() - start

print(f"Response time: {elapsed:.2f}s")
print(f"Generated tokens: {response.usage.completion_tokens}")
print(f"Tokens/sec: {response.usage.completion_tokens / elapsed:.1f}")
print(f"\nResponse:\n{response.choices[0].message.content[:500]}...")
print()

# Test 2: Multimodal input (image + text)
print("=== Test 2: Multimodal Input ===")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Read all the text in this image."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    max_tokens=1024
)
elapsed = time.time() - start

print(f"Response time: {elapsed:.2f}s")
print(f"Generated tokens: {response.usage.completion_tokens}")
print(f"Tokens/sec: {response.usage.completion_tokens / elapsed:.1f}")
print(f"\nResponse:\n{response.choices[0].message.content[:500]}...")
print()

# Test 3: Streaming response
print("=== Test 3: Streaming Response ===")
messages = [
    {
        "role": "user",
        "content": "Write a short story about a robot learning to paint."
    }
]

start = time.time()
stream = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    max_tokens=512,
    stream=True
)

first_token_time = None
token_count = 0

print("Streaming output:")
for chunk in stream:
    if chunk.choices[0].delta.content:
        if first_token_time is None:
            first_token_time = time.time()
            print(f"Time to first token: {first_token_time - start:.2f}s")
        
        print(chunk.choices[0].delta.content, end="", flush=True)
        token_count += 1

elapsed = time.time() - start
print(f"\n\nTotal time: {elapsed:.2f}s")
print(f"Tokens generated: ~{token_count}")
if first_token_time:
    decode_time = elapsed - (first_token_time - start)
    print(f"Decode throughput: ~{token_count / decode_time:.1f} tokens/sec")

print("\n=== All tests completed successfully! ===")
