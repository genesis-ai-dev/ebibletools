# Translation Benchmarks

This directory contains benchmarking tools for evaluating translation quality using different ContextQuery configurations.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Set up environment variables:**
   Copy the template and add your API keys:
   ```bash
   cp benchmarks/env.template .env
   # Then edit .env and add your actual OpenAI API key
   ```

## Usage

### Basic Usage

Run the benchmark with default settings (comparing 3 vs 5 examples):
```bash
python translation_benchmark.py
```

### Custom Configuration

Compare different numbers of examples:
```bash
python translation_benchmark.py --example-counts 1 3 5 10
```

Test with more target files and tests:
```bash
python translation_benchmark.py --num-target-files 5 --num-tests-per-file 10
```

Use a different source file:
```bash
python translation_benchmark.py --source-file your-source.txt
```

### All Options

```bash
python translation_benchmark.py \
  --api-key YOUR_API_KEY \
  --corpus-dir ../Corpus \
  --source-file eng-engULB.txt \
  --num-target-files 3 \
  --num-tests-per-file 8 \
  --example-counts 1 3 5 7 10
```

## Arguments

- `--api-key`: OpenAI API key (can also be set via OPENAI_API_KEY env var)
- `--corpus-dir`: Directory containing corpus files (default: ../Corpus)
- `--source-file`: Source file name (default: eng-engULB.txt)
- `--num-target-files`: Number of target files to test (default: 2)
- `--num-tests-per-file`: Number of tests per file (default: 5)
- `--example-counts`: Numbers of examples to compare (default: [3, 5])

## Output

The benchmark will output:
- Configuration summary
- Progress bars during testing
- Final results showing average similarity scores
- Performance comparison between different example counts 