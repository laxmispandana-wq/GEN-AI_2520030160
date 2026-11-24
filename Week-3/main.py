import os
from dotenv import load_dotenv

from usage_logger import UsageLogger
from llm_client import LLMClient

load_dotenv()


def main():

    print("=== LLM Monitoring System ===")
    print("1. Gemini")
    print("2. Ollama")

    choice = input("Choose provider (1 or 2): ").strip()

    if choice == "1":
        provider = "gemini"

    elif choice == "2":
        provider = "ollama"

    else:
        print("Invalid choice.")
        return

    usage_logger = UsageLogger()

    try:
        client = LLMClient(provider, usage_logger)

        prompt = input("Enter your prompt: ")

        result = client.generate(prompt)

        print("\n--- Response ---")
        print(result.text)

        print("\n--- Usage Details ---")
        print("Provider:", result.provider)
        print("Input Tokens:", result.input_tokens)
        print("Output Tokens:", result.output_tokens)
        print("Latency (ms):", result.latency_ms)
        print("Estimated Cost:", result.cost)

    except Exception as error:
        print("Error:", error)

    finally:
        usage_logger.close()


if __name__ == "__main__":
    main()