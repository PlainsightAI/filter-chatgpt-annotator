# Adding Other LLMs to ChatTag

## Overview

The `ChatTag` was specifically designed for the ChatGPT/OpenAI API. Adding support for other LLMs like Gemini (Google) or Nano Banana requires significant modifications to the system architecture.

## Implementation Complexity

### 1. **Current Architecture (ChatGPT)**

The current system is tightly coupled to the OpenAI API:

```python
# Current structure
class FilterChatgptAnnotatorConfig(FilterConfig):
    chatgpt_model: str = "gpt-4o-mini"
    chatgpt_api_key: str = ""
    # ... other ChatGPT-specific parameters

class FilterChatgptAnnotator(Filter):
    def _analyze_image_with_chatgpt(self, image, frame_id):
        # OpenAI API specific implementation
        response = self.client.chat.completions.create(...)
```

### 2. **Challenges for Adding Other LLMs**

#### **A. API Differences**

| Aspect | ChatGPT/OpenAI | Gemini | Nano Banana |
|--------|----------------|--------|-------------|
| **Authentication** | Simple API Key | OAuth2 + API Key | API Key + custom headers |
| **Request Format** | `chat.completions.create()` | `generateContent()` | Custom REST endpoint |
| **Response Format** | `response.choices[0].message.content` | `response.candidates[0].content` | Custom JSON |
| **Image Handling** | Base64 inline | Base64 or URLs | Multipart form-data |
| **Rate Limiting** | Tokens per minute | Requests per minute | Configurable |
| **Available Models** | gpt-4o, gpt-4o-mini | gemini-pro-vision | nano-banana-vision |

#### **B. Image Processing Differences**

```python
# ChatGPT (current)
image_b64 = base64.b64encode(buffer.getvalue()).decode()
response = self.client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": self.prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }]
)

# Gemini (example)
response = self.client.models.generate_content({
    "contents": [{
        "parts": [
            {"text": self.prompt_text},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}
        ]
    }]
})

# Nano Banana (example)
files = {"image": ("image.jpg", buffer.getvalue(), "image/jpeg")}
data = {"prompt": self.prompt_text}
response = requests.post(url, files=files, data=data, headers=headers)
```

### 3. **Implementation Complexity**

#### **Level 1: Minimal Refactoring (2-3 days)**
- Create abstract classes for different providers
- Implement adapters for each LLM
- Maintain compatibility with existing code

```python
# Proposed structure
class LLMProvider(ABC):
    @abstractmethod
    def analyze_image(self, image, prompt) -> tuple[dict, dict]:
        pass

class OpenAIProvider(LLMProvider):
    def analyze_image(self, image, prompt):
        # Current ChatGPT implementation

class GeminiProvider(LLMProvider):
    def analyze_image(self, image, prompt):
        # Gemini implementation

class FilterLLMAnnotator(Filter):
    def __init__(self, provider: LLMProvider):
        self.provider = provider
```

#### **Level 2: Complete Refactoring (1-2 weeks)**
- Redesign architecture to be provider-agnostic
- Implement unified configuration system
- Add tests for each provider
- Document differences between providers

### 4. **Provider-Specific Challenges**

#### **Google Gemini**
```python
# Gemini-specific complexities:
- OAuth2 authentication flow
- Different models for different tasks
- Rate limiting based on requests, not tokens
- Different response format
- Limited support for certain image types
```

#### **Nano Banana**
```python
# Nano Banana-specific complexities:
- Custom API (not standardized)
- Specific authentication headers
- Different image upload format
- Non-standardized response format
- Limited documentation
```

### 5. **Impact on Existing Code**

#### **Files that Would Need Modification:**

1. **`filter_chatgpt_annotator/filter.py`** (Main)
   - Refactor main class
   - Add provider system
   - Modify configuration validation

2. **`filter_chatgpt_annotator/__init__.py`**
   - Add imports for new providers
   - Maintain compatibility with current version

3. **Example scripts** (`scripts/`)
   - Update configurations
   - Add examples for each provider

4. **Documentation**
   - Update README
   - Create provider-specific guides

#### **Proposed Unified Configuration:**

```python
# Example of unified configuration
class FilterLLMAnnotatorConfig(FilterConfig):
    # Provider selection
    provider: str = "openai"  # "openai", "gemini", "nano_banana"
    
    # Provider-specific configs
    openai_config: Optional[OpenAIConfig] = None
    gemini_config: Optional[GeminiConfig] = None
    nano_banana_config: Optional[NanoBananaConfig] = None
    
    # Common configs
    prompt: str = ""
    output_schema: Dict[str, Any] = {}
    confidence_threshold: float = 0.9
```

### 6. **Effort Estimation**

| Task | Estimated Time | Complexity |
|------|----------------|------------|
| **Base architecture refactoring** | 3-5 days | High |
| **Gemini provider implementation** | 2-3 days | Medium |
| **Nano Banana provider implementation** | 3-4 days | High |
| **Testing and validation** | 2-3 days | Medium |
| **Documentation and examples** | 1-2 days | Low |
| **Migration and compatibility** | 1-2 days | Medium |

**Total: 12-19 days of development**

### 7. **Recommendations**

#### **Incremental Approach:**
1. **Phase 1:** Refactor to support multiple providers while maintaining compatibility
2. **Phase 2:** Implement Gemini (better documented)
3. **Phase 3:** Implement Nano Banana (more complex)

#### **Design Considerations:**
- Maintain compatibility with existing code
- Use factory pattern for provider creation
- Implement fallback between providers
- Add performance metrics per provider

#### **Alternatives:**
- **Keep separate:** Create `FilterGeminiAnnotator` and `FilterNanoBananaAnnotator` as independent classes
- **Plugin system:** Implement plugin system for providers
- **Unified wrapper:** Create wrapper that abstracts API differences

### 8. **Conclusion**

Adding support for other LLMs is **feasible but complex**, requiring:

- **Significant refactoring** of current architecture
- **Specific knowledge** of each API
- **Extensive testing** to ensure compatibility
- **Detailed documentation** for each provider

The **main complexity** lies in the fundamental differences between APIs, not just implementation details. Each provider has its own conventions, limitations, and characteristics that need to be carefully mapped and abstracted.
