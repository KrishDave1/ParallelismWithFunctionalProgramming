# Harnessing Functional Programming for Parallelism

> **Course:** Programming Languages (8th Semester)  
> **Author:** Krish Dave,   Aaditya Joshi    
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


## Benchmark Results & Code Explanations (8-core Apple Silicon)

### Problem 1: Parallel Merge Sort
**Explanation:** Merge sort is a divide-and-conquer algorithm. C++ uses `std::async` for recursively sorting halves in-place, yielding massive speedups. Haskell uses the `Eval` monad to spark parallel evaluations. Python uses `multiprocessing.Pool`, but because sorting requires pickling (serializing) the entire array to send to worker processes and unpickling the result back, the IPC overhead heavily outweighs the parallel computing benefits for smaller arrays. Python actually gets *slower* with 2/4 workers for small sizes, only slightly breaking even at larger sizes.

**Language Comparison (Sequential vs 4-Worker Parallel)**
| Input Size | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|---|---|---|---|---|---|---|
| 10K | 14.35 ms | 3.76 ms (3.81x) | 0.64 ms | 0.27 ms (2.37x) | 23.12 ms | 114.30 ms (0.20x) |
| 50K | 68.13 ms | 31.97 ms (2.13x) | 3.2 ms | 0.8 ms (4.00x) | 138.32 ms | 123.61 ms (1.12x) |
| 100K | 172.83 ms | 76.69 ms (2.25x) | 7.47 ms | 2.05 ms (3.64x) | 313.87 ms | 278.82 ms (1.13x) |

**1. Haskell Scaling (Time vs Chunks)**
| Input Size | Seq Time | 2 Chunks | 4 Chunks | 8 Chunks |
|---|---|---|---|---|
| 10K | 14.35 ms | 4.49 ms (3.20x) | 3.76 ms (3.81x) | 3.97 ms (3.62x) |
| 50K | 68.13 ms | 34.94 ms (1.95x) | 31.97 ms (2.13x) | 31.56 ms (2.16x) |
| 100K | 172.83 ms | 78.76 ms (2.19x) | 76.69 ms (2.25x) | 68.61 ms (2.52x) |

**2. C++ Scaling (Time vs Threads)**
| Input Size | Seq Time | 2 Threads | 4 Threads | 8 Threads |
|---|---|---|---|---|
| 10K | 0.64 ms | 0.28 ms (2.28x) | 0.27 ms (2.37x) | 0.27 ms (2.37x) |
| 50K | 3.20 ms | 0.90 ms (3.55x) | 0.80 ms (4.00x) | 0.80 ms (4.00x) |
| 100K | 7.47 ms | 2.42 ms (3.09x) | 2.05 ms (3.64x) | 1.74 ms (4.29x) |

**3. Python Scaling (Time vs Workers)**
| Input Size | Seq Time | 2 Workers | 4 Workers |
|---|---|---|---|
| 10K | 23.12 ms | 85.27 ms (0.27x) | 114.30 ms (0.20x) |
| 50K | 138.32 ms | 145.05 ms (0.95x) | 123.61 ms (1.12x) |
| 100K | 313.87 ms | 245.02 ms (1.28x) | 278.82 ms (1.13x) |

---

### Problem 2: Matrix Multiplication
**Explanation:** This is highly CPU-bound. C++ shows phenomenal sequential and parallel performance because the 2D `vector` loops maximize CPU cache locality. Haskell distributes chunked rows automatically across cores via the `Strategies` library, gaining decent speedups. Python suffers again from IPC serialization overhead when distributing matrix chunks to different processes, only achieving mild speedups at 128x128.

**Language Comparison (Sequential vs 4-Worker Parallel)**
| Size | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|---|---|---|---|---|---|---|
| 64x64 | 3.12 ms | 2.04 ms (1.53x) | 56 us | 78 us (0.72x) | 27.33 ms | 78.53 ms (0.35x) |
| 128x128 | 18.77 ms | 14.48 ms (1.30x) | 920 us | 232 us (3.97x) | 178.95 ms | 128.25 ms (1.40x) |

**1. Haskell Scaling (Time vs Chunks)**
| Size | Seq Time | 4 Chunks | 8 Chunks |
|---|---|---|---|
| 64x64 | 3.12 ms | 2.04 ms (1.53x) | 2.35 ms (1.33x) |
| 128x128 | 18.77 ms | 14.48 ms (1.30x) | 14.32 ms (1.31x) |
| 256x256 | 160.51 ms | 139.59 ms (1.15x) | 110.85 ms (1.45x) |

**2. C++ Scaling (Time vs Threads)**
| Size | Seq Time | 2 Threads | 4 Threads | 8 Threads |
|---|---|---|---|---|
| 64x64 | 56 us | 78 us (0.72x) | 78 us (0.72x) | 146 us (0.38x) |
| 128x128 | 920 us | 263 us (3.50x) | 232 us (3.97x) | 187 us (4.92x) |
| 256x256 | 4.02 ms | 2.31 ms (1.74x) | 1.44 ms (2.79x) | 857 us (4.69x) |

**3. Python Scaling (Time vs Workers)**
| Size | Seq Time | 2 Workers | 4 Workers |
|---|---|---|---|
| 64x64 | 27.33 ms | 80.10 ms (0.34x) | 78.53 ms (0.35x) |
| 128x128 | 178.95 ms | 150.74 ms (1.19x) | 128.25 ms (1.40x) |

---

### Problem 3: MapReduce Word Count
**Explanation:** C++ processes contiguous strings in memory, utilizing local `std::map`s on separate threads and merging them, yielding near-linear speedups. Python's `collections.Counter` handles local counts rapidly, and merging dictionaries is efficient enough that Python sees actual speedups (up to 2.4x) on large texts! Haskell's `String` is a linked list of characters, which introduces significant memory overhead and garbage collection pressure, causing parallel execution to sometimes be slower than sequential.

**Language Comparison (Sequential vs 4-Worker Parallel)**
| Words | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|---|---|---|---|---|---|---|
| 155K | 85.15 ms | 172.56 ms (0.49x) | 12.73 ms | 4.19 ms (3.04x) | 149.52 ms | 165.20 ms (0.91x) |
| 465K | 255.46 ms | 450.32 ms (0.57x) | 38.76 ms | 11.45 ms (3.39x) | 441.76 ms | 181.71 ms (2.43x) |

**1. Haskell Scaling (Time vs Chunks)**
| Words | Seq Time | 2 Chunks | 4 Chunks | 8 Chunks |
|---|---|---|---|---|
| 155K | 85.15 ms | 130.52 ms (0.65x) | 172.56 ms (0.49x) | 195.21 ms (0.44x) |
| 465K | 255.46 ms | 538.09 ms (0.47x) | 450.32 ms (0.57x) | 472.59 ms (0.54x) |

**2. C++ Scaling (Time vs Threads)**
| Words | Seq Time | 2 Threads | 4 Threads | 8 Threads |
|---|---|---|---|---|
| 155K | 12.73 ms | 7.68 ms (1.66x) | 4.19 ms (3.04x) | 3.89 ms (3.28x) |
| 465K | 38.76 ms | 21.61 ms (1.79x) | 11.45 ms (3.39x) | 11.28 ms (3.44x) |

**3. Python Scaling (Time vs Workers)**
| Words | Seq Time | 2 Workers | 4 Workers |
|---|---|---|---|
| 155K | 149.52 ms | 170.60 ms (0.88x) | 165.20 ms (0.91x) |
| 465K | 441.76 ms | 288.18 ms (1.53x) | 181.71 ms (2.43x) |

---

### Problem 4: Monte Carlo Pi Estimation
**Explanation:** This algorithm is "embarrassingly parallel" since the operations are purely math with no data dependencies. The only communication happens at the very end when accumulating hits. Because of this, Python scales extremely well (3x+ speedups) despite using separate processes. C++ scales linearly using lock-free atomics or async futures. Haskell cleanly parallelizes using `SplitMix` to deterministically divide the pseudo-random number generator states without shared mutation.

**Language Comparison (Sequential vs 4-Worker Parallel)**
| Samples | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|---|---|---|---|---|---|---|
| 100K | 1.10 ms | 0.40 ms (2.6x) | 1.50 ms | 366 us (4.11x) | 20.24 ms | 75.10 ms (0.27x) |
| 1M | 10.5 ms | 3.5 ms (3.0x) | 12.25 ms | 3.04 ms (4.03x) | 203.42 ms | 121.07 ms (1.68x) |
| 10M | 163 ms | 185.94 ms (0.88x) | 122.72 ms | 60.24 ms (2.04x) | 2.08 s | 660.33 ms (3.15x) |

**1. Haskell Scaling (Time vs Chunks - STM)**
| Samples | Seq Time | 2 Chunks | 4 Chunks |
|---|---|---|---|
| 1M | 10.5 ms | 6.5 ms (1.62x) | 3.5 ms (3.00x) |
| 10M | 163 ms | 106.37 ms (1.53x) | 185.94 ms (0.88x) |

**2. C++ Scaling (Time vs Threads - Async)**
| Samples | Seq Time | 2 Threads | 4 Threads | 8 Threads |
|---|---|---|---|---|
| 100K | 1.50 ms | 620 us (2.43x) | 366 us (4.11x) | 282 us (5.34x) |
| 1M | 12.25 ms | 5.95 ms (2.06x) | 3.04 ms (4.03x) | 1.85 ms (6.61x) |
| 10M | 122.72 ms | 59.41 ms (2.07x) | 60.24 ms (2.04x) | 52.99 ms (2.32x) |

**3. Python Scaling (Time vs Workers)**
| Samples | Seq Time | 2 Workers | 4 Workers | 8 Workers |
|---|---|---|---|---|
| 100K | 20.24 ms | 75.94 ms (0.27x) | 75.10 ms (0.27x) | 112.43 ms (0.18x) |
| 1M | 203.42 ms | 176.86 ms (1.15x) | 121.07 ms (1.68x) | 131.67 ms (1.54x) |
| 10M | 2.08 s | 1.32 s (1.57x) | 660.33 ms (3.15x) | 1.37 s (1.51x) |

---

### Problem 5: K-Means Clustering
**Explanation:** K-Means combines CPU-heavy distance metrics with frequent synchronization (recomputing centroids). C++ does not scale perfectly because thread synchronization barriers at each step dominate execution time for smaller point sets. Python actually gets *much slower* because it must serialize the entire list of points and send it to workers at *every single iteration* until convergence! Haskell's functional purity allows `parListChunk` to automatically parallelize assignment cleanly, but it still struggles with scaling overhead.

**Language Comparison (Sequential vs 4-Worker Parallel)**
| Points | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|---|---|---|---|---|---|---|
| 10K | 64.05 ms | 68.41 ms (0.94x) | 17.40 ms | 15.86 ms (1.10x) | 273.69 ms | 1.74 s (0.16x) |
| 50K | 772.82 ms | 431.82 ms (1.79x) | 15.34 ms | 10.14 ms (1.51x) | 1.90 s | 3.23 s (0.59x) |
| 100K | 1.29 s | 1.49 s (0.87x) | 22.82 ms | 49.01 ms (0.47x) | 2.38 s | 2.27 s (1.05x) |

**1. Haskell Scaling (Time vs Chunks)**
| Points | Seq Time | 2 Chunks | 4 Chunks | 8 Chunks |
|---|---|---|---|---|
| 10K | 64.05 ms | 79.12 ms (0.81x) | 68.41 ms (0.94x) | 70.35 ms (0.91x) |
| 50K | 772.82 ms | 437.40 ms (1.77x) | 431.82 ms (1.79x) | 429.91 ms (1.80x) |
| 100K | 1.29 s | 1.43 s (0.90x) | 1.49 s (0.87x) | 1.43 s (0.90x) |

**2. C++ Scaling (Time vs Threads)**
| Points | Seq Time | 2 Threads | 4 Threads | 8 Threads |
|---|---|---|---|---|
| 10K | 17.40 ms | 37.56 ms (0.46x) | 15.86 ms (1.10x) | 10.33 ms (1.68x) |
| 50K | 15.34 ms | 11.54 ms (1.33x) | 10.14 ms (1.51x) | 16.90 ms (0.91x) |
| 100K | 22.82 ms | 56.72 ms (0.40x) | 49.01 ms (0.47x) | 22.45 ms (1.02x) |

**3. Python Scaling (Time vs Workers)**
| Points | Seq Time | 2 Workers | 4 Workers |
|---|---|---|---|
| 10K | 273.69 ms | 1.28 s (0.21x) | 1.74 s (0.16x) |
| 50K | 1.90 s | 3.31 s (0.58x) | 3.23 s (0.59x) |
| 100K | 2.38 s | 2.91 s (0.82x) | 2.27 s (1.05x) |

---

### Problem 6: Numerical Integration (Trapezoidal Rule)
**Explanation:** This is another "embarrassingly parallel" domain-decomposition problem with zero shared state and zero data arrays to serialize. C++ achieves phenomenal, near-perfect linear scaling. Python likewise scales beautifully (up to 4x) because the workers are purely evaluating math functions over coordinate ranges rather than passing massive data structures back and forth. Haskell scales excellently through the `parList rdeepseq` strategy.

**Language Comparison (Sequential vs 4-Worker Parallel)**
| Function | N | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|---|---|---|---|---|---|---|---|
| sin(x) | 10M | 94.88 ms | 85.96 ms (1.10x) | 46.52 ms | 12.09 ms (3.85x) | 1.47 s | 436.29 ms (3.38x) |
| 4/(1+x²) | 10M | 73.43 ms | 38.32 ms (1.92x) | 13.90 ms | 4.07 ms (3.41x) | 1.96 s | 449.63 ms (4.37x) |
| x² | 10M | 99.24 ms | 55.55 ms (1.79x) | 13.88 ms | 3.59 ms (3.87x) | 1.38 s | 455.29 ms (3.04x) |

**1. Haskell Scaling (Time vs Chunks)**
| Function | N | Seq Time | 2 Chunks | 4 Chunks | 8 Chunks |
|---|---|---|---|---|---|
| sin(x) | 10M | 94.88 ms | 51.03 ms (1.86x) | 85.96 ms (1.10x) | 57.18 ms (1.66x) |
| 4/(1+x²) | 10M | 73.43 ms | 38.10 ms (1.93x) | 38.32 ms (1.92x) | 38.89 ms (1.89x) |
| x² | 10M | 99.24 ms | 58.54 ms (1.70x) | 55.55 ms (1.79x) | 53.96 ms (1.84x) |

**2. C++ Scaling (Time vs Threads)**
| Function | N | Seq Time | 2 Threads | 4 Threads | 8 Threads |
|---|---|---|---|---|---|
| sin(x) | 10M | 46.52 ms | 24.09 ms (1.93x) | 12.09 ms (3.85x) | 9.30 ms (5.00x) |
| 4/(1+x²) | 10M | 13.90 ms | 7.09 ms (1.96x) | 4.07 ms (3.41x) | 2.53 ms (5.50x) |
| x² | 10M | 13.88 ms | 6.97 ms (1.99x) | 3.59 ms (3.87x) | 2.25 ms (6.18x) |

**3. Python Scaling (Time vs Workers)**
| Function | N | Seq Time | 2 Workers | 4 Workers | 8 Workers |
|---|---|---|---|---|---|
| sin(x) | 10M | 1.47 s | 919.85 ms (1.60x) | 436.29 ms (3.38x) | 357.77 ms (4.12x) |
| 4/(1+x²) | 10M | 1.96 s | 759.30 ms (2.59x) | 449.63 ms (4.37x) | 407.86 ms (4.82x) |
| x² | 10M | 1.38 s | 602.69 ms (2.30x) | 455.29 ms (3.04x) | 608.97 ms (2.27x) |

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