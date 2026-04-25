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
│   │   │   └── ParallelPi.hs            # Problem 4: Async + STM
│   │   ├── KMeans/
│   │   │   ├── SequentialKMeans.hs      # Problem 5: Machine Learning baseline
│   │   │   └── ParallelKMeans.hs        # Problem 5: Strategies + parListChunk
│   │   ├── NumericalIntegration/
│   │   │   ├── SequentialIntegration.hs # Problem 6: Numerical Simulation baseline
│   │   │   └── ParallelIntegration.hs   # Problem 6: Domain decomposition
│   │   └── Bench/
│   │       └── Utils.hs                 # Benchmarking utilities
│   └── app/                        # Executable entry points
│       ├── Main.hs                      # Full benchmark suite
│       ├── BenchMergeSort.hs            # Standalone merge sort bench
│       ├── BenchMatMul.hs               # Standalone matrix bench
│       ├── BenchWordCount.hs            # Standalone word count bench
│       ├── BenchMonteCarlo.hs           # Standalone Monte Carlo bench
│       ├── BenchKMeans.hs               # Standalone K-Means bench
│       └── BenchIntegration.hs          # Standalone Integration bench
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
| **Domain Decomposition** | Integration (#6) | `parList`, independent domains | Embarrassingly parallel, super-linear speedups |
| **Algorithm vs Strategy** | K-Means (#5) | `using parListChunk` | Separate algorithm logic from parallel execution |

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
cabal run bench-kmeans -- +RTS -N4 -s
cabal run bench-integration -- +RTS -N4 -s
```

### Understanding RTS Flags
- `+RTS` — begin RTS (RunTime System) options
- `-N4` — use 4 OS threads (change to your core count)
- `-N` — auto-detect core count
- `-s` — print GC and spark statistics
- `-ls` — generate eventlog for ThreadScope visualization

---

## Benchmark Results & Code Explanations (8-core Apple Silicon, -N4)

### Problem 1: Parallel Merge Sort
**Implementation Overview:**
- **Haskell**: Uses the `Eval` monad (`rpar` and `rseq`) to spark parallel evaluation of the left and right sublists. A depth threshold limits spark creation to avoid overhead, falling back to sequential sorting at the leaves.
- **C++**: Leverages `std::async` to recursively spawn new threads for the left half while sorting the right half on the current thread. Mutations happen in-place using a shared temporary array, which is safe due to strict index boundaries.
- **Python**: Uses `multiprocessing.Pool` to bypass the Global Interpreter Lock (GIL). However, pickling (serializing) array chunks to send to worker processes introduces massive overhead, making parallel performance worse for small arrays.

| Input Size | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|-----------|------------|------------|---------|---------|-----------|-----------|
| 10K | 12.44 ms | **2.66 ms** (4.7x) | 0.64 ms | **0.28 ms** (2.3x) | 34 ms | 96 ms (0.36x) |
| 50K | 75.9 ms | **24.0 ms** (3.2x) | 3.2 ms | **0.9 ms** (3.6x) | 197 ms | 142 ms (1.4x) |
| 100K | 169.2 ms | **53.0 ms** (3.2x) | 6.7 ms | **1.5 ms** (4.5x) | 417 ms | 232 ms (1.8x) |
| 500K | 1.43 s | **518 ms** (2.8x) | 38 ms | **9.0 ms** (4.2x) | — | — |

### Problem 2: Matrix Multiplication
**Implementation Overview:**
- **Haskell**: Employs Data Parallelism via the `Strategies` library (`parMap rdeepseq`). We map the dot-product computation across the rows of the matrix, distributing chunked rows automatically across CPU cores.
- **C++**: Uses `std::thread` to divide the outer loop (rows) among threads. Pre-allocates a 2D `vector` and performs deeply nested loop iterations. Extreme cache locality makes this blazingly fast.
- **Python**: Uses `multiprocessing.Pool.map`. Since matrices must be pickled across the IPC boundary, the overhead is substantial, only achieving speedup at larger matrix dimensions.

| Size | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|------|------------|------------|---------|---------|-----------|-----------|
| 64×64 | 2.5 ms | **2.2 ms** (1.1x) | 130 μs | **114 μs** (1.1x) | 42 ms | 121 ms (0.3x) |
| 128×128 | 19 ms | **6.1 ms** (3.1x) | 620 μs | **670 μs** (0.9x) | 276 ms | 135 ms (2.0x) |
| 256×256 | 166 ms | **49 ms** (3.4x) | 4.2 ms | **1.1 ms** (4.0x) | — | — |

### Problem 3: MapReduce Word Count
**Implementation Overview:**
- **Haskell**: Utilizes the `Par` monad (`runPar` and `parMapM`) to guarantee deterministic parallelism. The text is chunked, counted locally, and reduced using `unionsWith (+)`. Overhead stems from Haskell's `String` being a linked list of characters.
- **C++**: Uses `std::async` to process string chunks into local `std::map`s, then reduces them sequentially at the end. In-place string mutation and contiguous memory make the counting highly efficient.
- **Python**: Uses `collections.Counter` locally within a multiprocessing pool. Python's C-backed dictionary (hash map) implementation is highly optimized, but inter-process string copying hurts scaling.

| Words | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|-------|------------|------------|---------|---------|-----------|-----------|
| 155K | 85 ms | 134 ms (0.6x) | 13 ms | **3.9 ms** (3.3x) | 47 ms | 88 ms (0.5x) |
| 465K | 256 ms | 414 ms (0.6x) | 39 ms | **12 ms** (3.3x) | 143 ms | 114 ms (1.3x) |

### Problem 4: Monte Carlo Pi Estimation
**Implementation Overview:**
- **Haskell**: Features `async` for future/promise-based concurrency and `STM` (Software Transactional Memory) with `TVar`s for safely accumulating results. Crucially, uses `SplitMix` to purely split PRNG states so threads don't share random seeds.
- **C++**: Uses `std::async` with local `std::mt19937` instances seeded by thread IDs. Atomic counters (`std::atomic<int>`) safely aggregate hits across threads without locking overhead.
- **Python**: Leverages `multiprocessing.Pool` with local `random.Random` instances. Communication overhead for a single integer result is minimal, so scaling is reasonable for large sample sizes.

| Samples | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|---------|------------|------------|---------|---------|-----------|-----------|
| 100K | 1.1 ms | **0.4 ms** (2.6x) | 1.2 ms | **0.4 ms** (3.0x) | 21 ms | 80 ms (0.3x) |
| 1M | 10.5 ms | **3.2 ms** (3.3x) | 12 ms | **3.6 ms** (3.4x) | 227 ms | 128 ms (1.8x) |
| 10M | 163 ms | **30 ms** (5.4x) | 124 ms | **32 ms** (3.9x) | 2.2 s | 798 ms (2.8x) |

### Problem 5: K-Means Clustering (Machine Learning)
**Implementation Overview:**
- **Haskell**: Defines custom algebraic data types (ADTs) for points and centroids. The parallel algorithm uses `using parListChunk` — the exact same pure functions as the sequential version, cleanly separating algorithm logic from execution strategy.
- **C++**: Accumulates cluster assignments into mutable vectors. Parallelization uses `std::thread` to chunk distance calculations across workers, joining threads at each convergence iteration.
- **Python**: Uses `multiprocessing.Pool` to chunk distance metrics. The centroid update phase remains sequential. Overhead from serializing points list at every iteration limits parallel gains.

| Points | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|--------|------------|------------|---------|---------|-----------|-----------|
| 10K | 76 ms | **41 ms** (1.8x) | 1.1 ms | 1.8 ms (0.6x) | 361 ms | 1.1 s (0.3x) |
| 50K | 429 ms | **407 ms** (1.1x) | 5.1 ms | 6.0 ms (0.9x) | 2.6 s | 2.6 s (1.0x) |
| 100K | 1.32 s | **1.22 s** (1.1x) | 14 ms | 16 ms (0.8x) | 3.3 s | 2.4 s (1.4x) |
| 200K | 4.08 s | **3.12 s** (1.3x) | 29 ms | 30 ms (1.0x) | — | — |

### Problem 6: Numerical Integration (Trapezoidal Rule)
**Implementation Overview:**
- **Haskell**: Implements a higher-order function taking the equation as a parameter. Parallelizes via Domain Decomposition: splits the interval into sub-intervals, evaluates them using `parList rdeepseq`, and computes a strict fold sum.
- **C++**: Creates `std::future`s mapping independent sub-domains to threads. Since there is zero shared state and mathematical operations map directly to CPU registers, speedups are phenomenal.
- **Python**: Decomposes the domain and maps function evaluations to a process pool. Embarrassingly parallel nature allows Python to overcome IPC bottlenecks for very large intervals.

| Function | N | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|----------|---|------------|------------|---------|---------|-----------|-----------|
| sin(x) | 1M | 50 ms | **50 ms** (1.0x) | 4.4 ms | **1.4 ms** (3.0x) | 199 ms | 126 ms (1.6x) |
| sin(x) | 10M | 226 ms | **39 ms** (5.8x) | 46 ms | **15 ms** (3.1x) | 2.0 s | 663 ms (3.0x) |
| 4/(1+x²) | 1M | 14 ms | **4.4 ms** (3.2x) | 1.4 ms | **486 μs** (2.9x) | 194 ms | 116 ms (1.7x) |
| 4/(1+x²) | 10M | 105 ms | **25 ms** (4.2x) | 14 ms | **4.1 ms** (3.4x) | 1.87 s | 570 ms (3.3x) |
| x² | 10M | 94 ms | **23 ms** (4.1x) | 14 ms | **3.8 ms** (3.7x) | 1.56 s | 461 ms (3.4x) |

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
9. `KMeans/SequentialKMeans.hs` — pure iteration, algebraic data types
10. `KMeans/ParallelKMeans.hs` — Strategy-based parallelism with parListChunk
11. `NumericalIntegration/SequentialIntegration.hs` — higher-order functions, strict folds
12. `NumericalIntegration/ParallelIntegration.hs` — domain decomposition, separation of concerns