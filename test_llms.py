import dspy
import config

def test_llm(name, model_name, api_key, test_prompt):
    """Test a single LLM model with the given prompt"""
    # Configure based on provider with proper settings
    if name == "gemini":
        # Try different formats for Gemini
        lm = dspy.LM(model=model_name, api_key=api_key, max_tokens=16000)
    elif name == "claude":
        lm = dspy.LM(
            model=f"anthropic/{model_name}", api_key=api_key, max_tokens=16000)
    elif name == "chatgpt":
        # Use correct temperature and max_tokens for reasoning models
        lm = dspy.LM(model=f"openai/{model_name}", api_key=api_key, temperature=1.0, max_tokens=16000)
    else:
        lm = dspy.LM(model=model_name, api_key=api_key, max_tokens=1000)

    with dspy.context(lm=lm):
        response = lm(test_prompt)
    print(f"Raw response: {response}")


def main():
    test_prompt = "This is a test prompt. respond with your name"

    models = [
        ("gemini", config.GEMINI_MODEL, config.GEMINI_API_KEY),
        ("claude", config.CLAUDE_MODEL, config.CLAUDE_API_KEY),
        ("chatgpt", config.CHATGPT_MODEL, config.CHATGPT_API_KEY)
    ]

    for name, model, api_key in models:
        print(f"\nTesting {name} ({model}):")
        test_llm(name, model, api_key, test_prompt)

if __name__ == "__main__":
    main()