# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an AIGC (AI-Generated Content) portfolio repository containing three major projects demonstrating LLM optimization and RAG applications. Total codebase: ~10,000 lines across C++17 and Python.

**Projects:**
1. **llm-inference-engine** - High-performance C++17 LLM inference engine with KV Cache, INT8 quantization, and SIMD optimizations
2. **code-qa-rag-system** - RAG-based code Q&A system using LangChain, ChromaDB, and OpenAI
3. **mini_projects** - Four rapid prototype projects (Mini-RAG, quantization tool, prompt optimizer, benchmark tool)

## Build & Development Commands

### C++ Inference Engine (llm-inference-engine)

**Build:**
```bash
cd llm-inference-engine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Run tests:**
```bash
# From build directory
./test_kv_cache          # Test KV Cache implementation
./test_quantization      # Test INT8/INT4 quantization
./benchmark              # Performance benchmarks
```

**Build options:**
- Release build with optimizations: `cmake -DCMAKE_BUILD_TYPE=Release ..`
- Aggressive optimization: `cmake -DCMAKE_CXX_FLAGS="-O3 -march=native" ..`
- AVX2 support is auto-detected and enabled if available

**Key files:**
- `cpp/include/kv_cache.h` - KV Cache interface
- `cpp/include/quantization.h` - Quantization interface
- `cpp/src/kv_cache.cpp` - KV Cache implementation
- `cpp/src/quantization.cpp` - Quantization with SIMD optimizations
- `CMakeLists.txt` - Build configuration

### RAG Q&A System (code-qa-rag-system)

**Setup:**
```bash
cd code-qa-rag-system
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

**Index a codebase:**
```bash
python code_indexer.py  # Indexes current directory
```

**Run the system:**
```bash
python app.py           # Launch Gradio web interface (port 7860)
python example.py       # Interactive CLI examples
```

**Key files:**
- `code_indexer.py` - Code loading and indexing with tree-sitter
- `qa_engine.py` - RAG query engine
- `code_loader.py` - Multi-language code file loader
- `app.py` - Gradio web interface
- `config.py` - All configuration (models, chunk sizes, prompts)

**Dependencies:**
- langchain, langchain-openai, openai
- chromadb (vector database)
- gradio (web UI)
- tree-sitter, pygments (code parsing)

### Mini Projects (mini_projects)

Each mini project is standalone:
```bash
cd mini_projects/mini_rag && python mini_rag.py
cd mini_projects/quantization_tool && python quantizer.py
cd mini_projects/prompt_optimizer && python prompt_optimizer.py
cd mini_projects/benchmark_tool && python benchmark.py
```

## Theory and Learning Resources

This repository includes comprehensive theoretical documentation in the `docs/` directory to support deep understanding of AIGC technologies.

### Theory Documents

**Core Concepts:**
1. **[LLM Fundamentals](docs/01_LLM_Fundamentals.md)** - Transformer architecture, self-attention mechanism, autoregressive generation, model scaling
2. **[Inference Optimization](docs/02_Inference_Optimization.md)** - KV Cache theory, quantization mathematics, SIMD vectorization, performance analysis
3. **[RAG System Theory](docs/03_RAG_System_Theory.md)** - Vector embeddings, retrieval algorithms, evaluation metrics, advanced RAG techniques
4. **[Prompt Engineering](docs/04_Prompt_Engineering.md)** - Zero-shot/Few-shot, Chain-of-Thought, ReAct, optimization strategies
5. **[Learning Roadmap](docs/05_Learning_Roadmap.md)** - Complete 3-month learning path from basics to advanced

**Advanced Topics:**
6. **[Model Training and Finetuning](docs/06_Model_Training_and_Finetuning.md)** - LoRA, QLoRA, RLHF, DPO, instruction tuning, distributed training strategies
7. **[LLM Evaluation](docs/07_LLM_Evaluation.md)** - MMLU, HumanEval, GSM8K benchmarks, LLM-as-Judge, hallucination detection, human evaluation
8. **[Agent Systems](docs/08_Agent_Systems.md)** - ReAct framework, function calling, tool use, planning strategies, memory management, multi-agent collaboration
9. **[Multimodal AI](docs/09_Multimodal_AI.md)** - CLIP, LLaVA, GPT-4V, Stable Diffusion, ControlNet, Whisper, vision-language models
10. **[Production Deployment](docs/10_Production_Deployment.md)** - vLLM, TensorRT-LLM, continuous batching, monitoring, cost optimization, high availability
11. **[Safety and Alignment](docs/11_Safety_and_Alignment.md)** - Prompt injection defense, toxicity detection, Constitutional AI, privacy protection, red teaming

**Knowledge Coverage Analysis:**
- **[Knowledge Gap Analysis](docs/00_Knowledge_Gap_Analysis.md)** - Complete analysis of AIGC knowledge system coverage (93% complete), learning paths, and next steps

### When to Consult Theory Docs

**Before implementing:**
- Read [LLM Fundamentals](docs/01_LLM_Fundamentals.md) to understand why KV Cache provides 20x speedup
- Review [Inference Optimization](docs/02_Inference_Optimization.md) for quantization trade-offs (INT8 vs INT4)
- Check [RAG System Theory](docs/03_RAG_System_Theory.md) for chunking strategies and retrieval methods
- Consult [Agent Systems](docs/08_Agent_Systems.md) before building autonomous agents with tool use

**When optimizing:**
- [Inference Optimization](docs/02_Inference_Optimization.md) explains SIMD optimization techniques and memory bandwidth bottlenecks
- [RAG System Theory](docs/03_RAG_System_Theory.md) covers hybrid search and re-ranking strategies
- [Prompt Engineering](docs/04_Prompt_Engineering.md) provides A/B testing frameworks and evaluation metrics
- [Production Deployment](docs/10_Production_Deployment.md) covers vLLM setup, monitoring, and cost optimization

**When training/fine-tuning:**
- [Model Training and Finetuning](docs/06_Model_Training_and_Finetuning.md) explains LoRA, QLoRA, RLHF, and DPO with full code examples
- [LLM Evaluation](docs/07_LLM_Evaluation.md) provides benchmarking methods and hallucination detection techniques
- [Safety and Alignment](docs/11_Safety_and_Alignment.md) covers alignment techniques and security best practices

**When building multimodal applications:**
- [Multimodal AI](docs/09_Multimodal_AI.md) covers CLIP for image-text matching, Stable Diffusion for generation, Whisper for speech

**For learning:**
- Start with [Learning Roadmap](docs/05_Learning_Roadmap.md) for structured progression
- Consult [Knowledge Gap Analysis](docs/00_Knowledge_Gap_Analysis.md) to understand coverage and choose your path
- Each theory doc includes mathematical derivations, code examples, and practical guidance
- Cross-references between docs for deep dives into specific topics

### Key Theoretical Insights

**LLM Inference:**
- Attention complexity: O(n²) in sequence length - this is why long contexts are expensive
- Prefill vs Decode: Prefill is compute-bound (fast), Decode is memory-bound (slow)
- KV Cache trades 1GB memory for 20x speedup by avoiding recomputation

**Quantization:**
- INT8: `scale = max(|W|) / 127`, provides 4x compression with <1% accuracy loss
- Group quantization: Per-group scales improve SQNR from 35dB to 42dB
- Memory bandwidth: INT8 reads 2x faster than FP16, directly improving decode speed

**RAG Retrieval:**
- Embedding similarity: Cosine similarity in 1536-dimensional space captures semantic meaning
- Chunk size trade-off: Larger chunks (1000+ chars) preserve context, smaller chunks (500 chars) improve precision
- Top-K selection: 3-5 documents optimal for most tasks (more adds noise, fewer miss information)

**SIMD Optimization:**
- AVX2: 256-bit registers = 8x float32 parallelism (theoretical 8x, practical 3-4x due to memory bandwidth)
- Memory alignment: 32-byte aligned loads are 2x faster than unaligned
- FMA instruction: Fused multiply-add reduces latency and improves accuracy

## Architecture & Design Patterns

### LLM Inference Engine Architecture

**Core optimization strategy:** Three-layered optimization approach
1. **KV Cache** - Caches key/value matrices to avoid O(n²) recomputation during autoregressive decoding
2. **INT8 Quantization** - Symmetric per-channel quantization with configurable group sizes (32/64/128)
3. **SIMD Vectorization** - AVX2 instructions for 8-way parallel float operations

**Memory layout:** Contiguous arrays for cache-friendly access patterns. KV Cache uses pre-allocated arrays with O(1) updates.

**Key design choices:**
- Static library + shared library build for flexibility
- Optional Python bindings via pybind11
- Header-only utils for inlining critical paths
- Separate test executables linked against static lib

### RAG System Architecture

**Pipeline flow:**
```
User Query → Query Processing → Vector Retrieval (ChromaDB) → Context Assembly → LLM Generation → Response
```

**Code chunking strategy:**
- Uses RecursiveCharacterTextSplitter with code-aware separators
- Priority: paragraph breaks → class definitions → function definitions → lines
- Chunk size: 1000 chars with 200 char overlap to preserve context
- Metadata tracking: file path, language, line numbers

**Retrieval approach:**
- Embeddings: OpenAI text-embedding-3-small (1536 dimensions)
- Search: Cosine similarity (default) or MMR for diversity
- Top-K: Default 3 documents, configurable in config.py

**LLM configuration:**
- Default model: gpt-4o-mini (cost-efficient)
- Temperature: 0 for deterministic outputs
- Custom prompts in config.py for Q&A, code review, code explanation

### Mini Projects Design

**Philosophy:** Rapid prototyping (1-2 days each) to validate concepts before full implementation. Each project is self-contained with minimal dependencies.

## Common Development Workflows

### Adding new quantization methods

1. Add quantization logic in `llm-inference-engine/cpp/src/quantization.cpp`
2. Update interface in `cpp/include/quantization.h`
3. Add test case in `cpp/tests/test_quantization.cpp`
4. Rebuild and run: `cd build && make && ./test_quantization`

### Modifying RAG retrieval behavior

1. Edit retrieval parameters in `code-qa-rag-system/config.py`:
   - `RETRIEVAL_TOP_K` - number of documents to retrieve
   - `RETRIEVAL_SEARCH_TYPE` - "similarity" or "mmr"
   - `CHUNK_SIZE` / `CHUNK_OVERLAP` - chunking parameters
2. For prompt changes, modify `QA_PROMPT_TEMPLATE` in config.py
3. Reindex if chunking changed: `python code_indexer.py`

### Running performance benchmarks

**C++ engine:**
```bash
cd llm-inference-engine/build
./benchmark  # Outputs timing comparisons for KV Cache, quantization, SIMD
```

**Python benchmarks:**
```bash
cd mini_projects/benchmark_tool
python benchmark.py --model gpt-3.5-turbo --metric all
```

## Important Technical Details

### C++ Inference Engine

- **Requires C++17** - Uses structured bindings, std::optional, if constexpr
- **AVX2 optional** - Code auto-falls back to scalar if unavailable
- **Memory alignment** - SIMD code requires 32-byte alignment for optimal performance
- **No external dependencies** - Pure C++17 stdlib (except optional pybind11)

### RAG System

- **OpenAI API required** - Set OPENAI_API_KEY environment variable
- **Vector DB persistence** - ChromaDB stores data in `data/vector_db/` directory
- **Supported languages** - Python, C/C++, Java, JavaScript/TypeScript, Go, Rust, Swift, Kotlin, Ruby, PHP
- **Ignored directories** - Automatically skips .git, node_modules, venv, build, etc. (see config.py)

### Performance Characteristics

**C++ Engine:**
- KV Cache: 20x speedup for autoregressive decoding
- INT8 quantization: 4x memory reduction, 2-3x inference speedup
- Combined optimizations: ~30x overall speedup

**RAG System:**
- Retrieval accuracy: 85%+ (Top-5)
- Response time: <1s for most queries (P95)
- Scales to 100k+ lines of code

## Configuration Files

- `llm-inference-engine/CMakeLists.txt` - C++ build configuration
- `code-qa-rag-system/config.py` - All RAG system settings (models, prompts, chunking)
- `code-qa-rag-system/requirements.txt` - Python dependencies
- `.gitignore` - Ignores build artifacts, Python cache, vector DB, credentials

## Testing

**C++ tests:**
- Unit tests for KV Cache and quantization
- Benchmark suite measuring TTFT, throughput, memory usage
- All tests in `llm-inference-engine/cpp/tests/`

**Python tests:**
- Example usage in `code-qa-rag-system/example.py`
- Interactive testing via web UI: `python app.py`

## Environment Setup

**C++ development:**
- GCC 7+ or Clang 5+ with C++17 support
- CMake 3.15+
- Optional: CPU with AVX2 support for SIMD

**Python development:**
- Python 3.8+
- Virtual environment recommended
- OpenAI API key required for RAG system and mini projects

## Project Structure Context

```
.
├── llm-inference-engine/      # C++17 inference engine
│   ├── cpp/
│   │   ├── include/           # Header files
│   │   ├── src/               # Implementation
│   │   └── tests/             # Test executables
│   ├── CMakeLists.txt         # Build configuration
│   └── QUICKSTART.md          # Detailed setup guide
│
├── code-qa-rag-system/        # Python RAG system
│   ├── code_indexer.py        # Indexing pipeline
│   ├── qa_engine.py           # Query engine
│   ├── code_loader.py         # File loading
│   ├── app.py                 # Web interface
│   ├── config.py              # Configuration
│   ├── requirements.txt       # Dependencies
│   └── examples/              # Sample projects
│
└── mini_projects/             # Prototype projects
    ├── mini_rag/              # Simplified RAG (539 lines)
    ├── quantization_tool/     # INT8/INT4 quantizer (642 lines)
    ├── prompt_optimizer/      # Prompt engineering (712 lines)
    └── benchmark_tool/        # Performance testing (671 lines)
```

## Special Notes

- **ChromaDB persistence:** Vector database is stored locally. Delete `data/vector_db/` to reset.
- **Reindexing:** Required after modifying chunking parameters or adding new code files.
- **SIMD debugging:** If AVX2 issues occur, disable with `-DCMAKE_CXX_FLAGS=""` and rebuild.
- **API costs:** RAG system makes OpenAI API calls. Monitor usage via OpenAI dashboard.

## Git Commit Guidelines

**IMPORTANT: When creating git commits in this repository:**

- ❌ **DO NOT** add "Generated with Claude Code" attribution
- ❌ **DO NOT** add "Co-Authored-By: Claude" footer
- ✅ **DO** write clean, concise commit messages
- ✅ **DO** follow conventional commits format (e.g., `docs:`, `feat:`, `fix:`)

**Example of correct commit message:**
```
docs: add LLM evaluation benchmarks

- Add MMLU and HumanEval benchmark guides
- Include hallucination detection methods
- Update knowledge coverage analysis
```

**Repository owner preference:** Keep commit history clean without AI tool attribution.
