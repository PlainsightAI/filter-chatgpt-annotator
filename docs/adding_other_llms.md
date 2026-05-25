# Adding more LLM providers

`FilterChatTag` is built on top of [LangChain](https://python.langchain.com/)'s [`init_chat_model`](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html), so adding a new provider does **not** require any code changes in this filter. You install the matching `langchain-<provider>` package and point `FILTER_CHATTAG_MODEL` at it.

## Providers shipped by default

The following four are installed by `make install` / `pip install filter-chattag` so they work out of the box:

| Provider | `FILTER_CHATTAG_MODEL` example | Credential env var |
| --- | --- | --- |
| OpenAI | `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Google Gemini | `google_genai:gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Anthropic Claude | `anthropic:claude-3-5-sonnet-latest` | `ANTHROPIC_API_KEY` |
| Ollama (local) | `ollama:llava` | `OLLAMA_HOST` (default `http://localhost:11434`) |

## Adding a different provider

LangChain supports many more (AWS Bedrock, Azure OpenAI, Mistral, Cohere, Fireworks, Together, Groq, …). For each:

1. **Install** the LangChain integration:
   ```bash
   pip install langchain-<provider>
   ```
2. **Find the provider prefix** in [LangChain's provider list](https://python.langchain.com/docs/integrations/chat/).
3. **Set the env vars** the provider expects (LangChain reads them natively).
4. **Use it**:
   ```bash
   export FILTER_CHATTAG_MODEL=<provider>:<model>
   make run
   ```

Example with AWS Bedrock + Anthropic-on-Bedrock:

```bash
pip install langchain-aws
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export FILTER_CHATTAG_MODEL=bedrock_converse:anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Vision support is the only hard requirement

`FilterChatTag` always sends multimodal `HumanMessage` content (text + base64-encoded image). Models without vision support will refuse the request. Verify the model you pick supports image input — most "latest" Claude, GPT-4o, Gemini, and `llava`/`llama3.2-vision` Ollama models do.

## Structured output

`FilterChatTag` calls `model.with_structured_output(Pydantic, include_raw=True)`, which works on the four shipped providers via their native mechanism (tool-calling for OpenAI / Anthropic / Gemini, JSON-mode for Ollama). If you bring a provider whose `with_structured_output` is not yet implemented in `langchain-*`, the filter will raise on first invocation — open an issue and we can add a JSON-parsing fallback.

## What used to live here

Before v1.0.0 this document described the substantial effort required to add a second provider when the filter was directly coupled to the `openai` SDK. That coupling is gone — the LangChain abstraction is exactly what made this rewrite possible. See [MIGRATION.md](../MIGRATION.md) for the v1.0.0 rebrand details.
