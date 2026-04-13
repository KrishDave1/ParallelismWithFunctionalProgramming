# Harnessing Functional Programming for Parallelism

> **Course:** Programming Languages (8th Semester)  
> **Author:** Krish Dave  
> **Language:** Haskell (GHC 9.6.7) — compared with C++ and Python

---

## Project Overview

This project explores how functional programming constructs enable safe, composable, and efficient parallel programming. We implement four computational problems in Haskell using different parallelism mechanisms, and compare them with equivalent imperative implementations in C++ and Python.

### Central Thesis
> **Purity (lack of side effects) makes parallelism safe. Higher-order functions and lazy evaluation make it composable. Haskell provides multiple parallelism abstractions at different granularity levels.**

---

## Project Structure

```
Project/
├── haskell/                        # All Haskell implementations
│   ├── haskell-parallelism.cabal   # Build configuration
│   ├── src/                        # Library source code
│   │   ├── Sorting/
│   │   │   ├── SequentialMergeSort.hs   # Problem 1: Sequential baseline
│   │   │   └── ParallelMergeSort.hs     # Problem 1: Eval monad + Strategies
│   │   ├── Matrix/
│   │   │   ├── SequentialMatMul.hs      # Problem 2: Sequential baseline
│   │   │   └── ParallelMatMul.hs        # Problem 2: parMap + chunking
│   │   ├── MapReduce/
│   │   │   ├── SequentialWordCount.hs   # Problem 3: Sequential baseline
│   │   │   └── ParallelWordCount.hs     # Problem 3: Par monad
│   │   ├── MonteCarlo/
│   │   │   ├── SequentialPi.hs          # Problem 4: Sequential baseline
│   │   │   └── ParallelPi.hs           # Problem 4: Async + STM
│   │   └── Bench/
│   │       └── Utils.hs                 # Benchmarking utilities
│   └── app/                        # Executable entry points
│       ├── Main.hs                      # Full benchmark suite
│       ├── BenchMergeSort.hs            # Standalone merge sort bench
│       ├── BenchMatMul.hs               # Standalone matrix bench
│       ├── BenchWordCount.hs            # Standalone word count bench
│       └── BenchMonteCarlo.hs           # Standalone Monte Carlo bench
├── cpp/                            # C++ imperative comparisons
├── python/                         # Python imperative comparisons
└── results/                        # Benchmark results and charts
```

---

## FP Concepts Demonstrated

| FP Concept | Problem | Haskell Mechanism | Why It Matters for Parallelism |
|-----------|---------|------------------|-------------------------------|
| **Purity / Referential Transparency** | All | Default in Haskell | No shared mutable state → no data races |
| **Higher-Order Functions** | MapReduce (#3) | `map`, `fold`, `filter`, `parMap` | Composable parallelism as function arguments |
| **Lazy Evaluation** | Merge Sort (#1) | Default + `force` for parallel | Sparks are cheap (thunks), but need explicit forcing |
| **Immutable Data Structures** | All | Lists, Maps | Safe sharing across threads without copying |
| **Eval Monad + Strategies** | Merge Sort (#1) | `rpar`, `rseq`, `runEval` | Semi-explicit task parallelism |
| **Data Parallelism** | Matrix (#2) | `parMap`, `parList`, `using` | Separate algorithm from parallelism strategy |
| **Par Monad** | MapReduce (#3) | `runPar`, `parMapM`, IVars | Deterministic guaranteed parallelism |
| **Futures/Promises** | Monte Carlo (#4) | `async`, `mapConcurrently` | Concurrent computation with result collection |
| **STM** | Monte Carlo (#4) | `TVar`, `atomically` | Composable atomic operations, no deadlocks |
| **Splittable PRNG** | Monte Carlo (#4) | `SplitMix`, `splitSMGen` | Lock-free parallel random number generation |

---

## Quick Start

### Prerequisites
- [GHCup](https://www.haskell.org/ghcup/) (installs GHC + Cabal)
- GHC 9.6+ and Cabal 3.10+

### Build & Run

```bash
cd haskell/

# Build everything
cabal build

# Run full benchmark suite (4 cores, with RTS stats)
cabal run haskell-parallelism -- +RTS -N4 -s

# Run individual benchmarks
cabal run bench-mergesort -- +RTS -N8 -s
cabal run bench-matmul -- +RTS -N4 -s
cabal run bench-wordcount -- +RTS -N4 -s
cabal run bench-montecarlo -- +RTS -N8 -s
```

### Understanding RTS Flags
- `+RTS` — begin RTS (RunTime System) options
- `-N4` — use 4 OS threads (change to your core count)
- `-N` — auto-detect core count
- `-s` — print GC and spark statistics
- `-ls` — generate eventlog for ThreadScope visualization

---

## Benchmark Results (8-core Apple Silicon, -N4)

### Problem 1: Parallel Merge Sort

| Input Size | Sequential | Parallel (depth=4) | Speedup |
|-----------|-----------|-------------------|---------|
| 10,000 | 12.44 ms | 2.66 ms | **4.67x** |
| 50,000 | 75.9 ms | 23.99 ms | **3.16x** |
| 100,000 | 169.25 ms | 52.58 ms | **3.22x** |
| 500,000 | 1.433 s | 517.91 ms | **2.77x** |

### Problem 2: Matrix Multiplication

| Size | Sequential | Parallel (parMap) | Speedup |
|------|-----------|------------------|---------|
| 64×64 | 2.48 ms | 2.31 ms | 1.07x |
| 128×128 | 19.11 ms | 6.09 ms | **3.14x** |
| 256×256 | 166.31 ms | 49.02 ms | **3.39x** |

### Problem 3: MapReduce Word Count

| Words | Sequential | Parallel (8 chunks) | Speedup |
|-------|-----------|-------------------|---------|
| 155K | 85.4 ms | 133.91 ms | 0.64x |
| 465K | 255.94 ms | 414.22 ms | 0.62x |

*Note: Parallel overhead dominates for string-heavy workloads. Analysis in report.*

### Problem 4: Monte Carlo Pi Estimation

| Samples | Sequential | Async (4 workers) | Speedup |
|---------|-----------|-------------------|---------|
| 100K | 1.06 ms | 407 μs | **2.60x** |
| 1M | 10.52 ms | 3.16 ms | **3.33x** |
| 10M | 162.51 ms | 30.29 ms | **5.37x** |

### Spark Statistics (Full Run)
```
SPARKS: 585 (478 converted, 0 overflowed, 0 dud, 45 GC'd, 62 fizzled)
```
- **81.7% conversion rate** — most sparks successfully executed in parallel
- Productivity: 58.2% of total user, 53.2% of total elapsed

---

## How to Read the Code

Each source file is extensively documented with:
1. **PURPOSE** — what the module does and why
2. **KEY FP CONCEPTS** — which functional programming ideas are demonstrated
3. **COMPARISON WITH IMPERATIVE** — how the equivalent C++/Python would look
4. **IMPLEMENTATION WALKTHROUGH** — step-by-step explanation of the code
5. **PERFORMANCE NOTES** — why certain design choices affect performance

Start reading in this order:
1. `Sorting/SequentialMergeSort.hs` — simplest, introduces FP basics
2. `Sorting/ParallelMergeSort.hs` — adds parallelism with Eval monad
3. `Matrix/SequentialMatMul.hs` — higher-order functions for numerics
4. `Matrix/ParallelMatMul.hs` — data parallelism with Strategies
5. `MapReduce/SequentialWordCount.hs` — the MapReduce pattern in FP
6. `MapReduce/ParallelWordCount.hs` — Par monad for guaranteed parallelism
7. `MonteCarlo/SequentialPi.hs` — pure RNG and strict accumulators
8. `MonteCarlo/ParallelPi.hs` — Async, STM, and splittable PRNGs