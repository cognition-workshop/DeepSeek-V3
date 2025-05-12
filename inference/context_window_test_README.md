# DeepSeek-V3 Context Window Testing Application

This application allows you to test DeepSeek-V3's performance across different context lengths, including implementation of the "Needle In A Haystack" (NIAH) test.

## Installation

The application uses the existing DeepSeek-V3 inference code, so no additional installation is required beyond the normal DeepSeek-V3 setup.

## Usage

The application provides two main testing modes:

1. **NIAH Test**: Tests the model's ability to retrieve specific information (a "needle") placed within a larger context (the "haystack").
2. **Custom Text Test**: Tests the model's performance with custom text input.

### NIAH Test

Run a single NIAH test with a specific context length:

```bash
python context_window_test.py \
    --ckpt-path /path/to/model/checkpoint \
    --config /path/to/config.json \
    --test-type niah \
    --context-length 16k \
    --needle-text "The capital of France is Paris, and it is known for the Eiffel Tower."
```

Run a benchmark across all supported context lengths:

```bash
python context_window_test.py \
    --ckpt-path /path/to/model/checkpoint \
    --config /path/to/config.json \
    --test-type niah \
    --benchmark
```

### Custom Text Test

Test with text from a file:

```bash
python context_window_test.py \
    --ckpt-path /path/to/model/checkpoint \
    --config /path/to/config.json \
    --test-type custom \
    --input-file /path/to/input.txt \
    --prompt "Summarize the above information concisely."
```

Test with provided text:

```bash
python context_window_test.py \
    --ckpt-path /path/to/model/checkpoint \
    --config /path/to/config.json \
    --test-type custom \
    --input-text "This is some test text to process." \
    --prompt "What is this text about?"
```

## Parameters

### General Parameters

- `--ckpt-path`: Path to the model checkpoint directory (required)
- `--config`: Path to the model configuration file (required)
- `--test-type`: Type of test to run (`niah` or `custom`, default: `niah`)
- `--max-new-tokens`: Maximum number of new tokens to generate (default: 100)
- `--temperature`: Temperature for sampling (default: 0.7)

### NIAH Test Parameters

- `--context-length`: Context length for NIAH test (`4k`, `16k`, `32k`, `64k`, `128k`, default: `16k`)
- `--needle-text`: Text to use as the needle in NIAH test
- `--benchmark`: Run benchmark across all context lengths

### Custom Text Parameters

- `--input-file`: Path to input text file
- `--input-text`: Custom input text
- `--prompt`: Prompt to append to the input text (default: "Summarize the above information concisely.")

## Examples

### Basic NIAH Test

```bash
python context_window_test.py \
    --ckpt-path ~/model/deepseek-v3 \
    --config ~/model/deepseek-v3/config.json \
    --test-type niah \
    --context-length 32k
```

### NIAH Benchmark

```bash
python context_window_test.py \
    --ckpt-path ~/model/deepseek-v3 \
    --config ~/model/deepseek-v3/config.json \
    --test-type niah \
    --benchmark
```

### Custom Text from File

```bash
python context_window_test.py \
    --ckpt-path ~/model/deepseek-v3 \
    --config ~/model/deepseek-v3/config.json \
    --test-type custom \
    --input-file ~/data/long_text.txt
```

## Performance Metrics

The application reports the following performance metrics:

- **NIAH Test**:
  - Success rate (whether the model correctly retrieves the needle)
  - Average inference time
  - Response for each test run

- **Custom Text Test**:
  - Context length (number of tokens)
  - Inference time
  - Tokens per second
  - Model's response
