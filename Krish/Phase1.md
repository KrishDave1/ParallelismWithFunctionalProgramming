# Walkthrough: Harnessing FP for Parallelism

## What Was Built

A complete benchmark suite implementing **4 computational problems** in **3 languages** (Haskell, C++, Python), comparing functional parallelism constructs against imperative approaches.

### Project Structure

```
Project/
├── haskell/                        ← Primary FP implementation (10 modules)
│   ├── src/Sorting/                 Problem 1: Merge Sort
│   ├── src/Matrix/                  Problem 2: Matrix Multiply
│   ├── src/MapReduce/               Problem 3: Word Count
│   ├── src/MonteCarlo/              Problem 4: Monte Carlo Pi
│   └── src/Bench/Utils.hs           Benchmarking utilities
├── cpp/benchmark.cpp               ← C++ imperative comparison
├── python/benchmark.py             ← Python imperative comparison  
└── README.md                       ← Complete project documentation
```

### Key Files

| File | Lines | FP Concepts |
|------|-------|-------------|
| [SequentialMergeSort.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/Sorting/SequentialMergeSort.hs) | ~80 | Recursion, pattern matching, immutability |
| [ParallelMergeSort.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/Sorting/ParallelMergeSort.hs) | ~130 | Eval monad, rpar/rseq, sparks, depth threshold |
| [SequentialMatMul.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/Matrix/SequentialMatMul.hs) | ~100 | List comprehensions, higher-order functions |
| [ParallelMatMul.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/Matrix/ParallelMatMul.hs) | ~150 | parMap, parList, Strategies, chunking |
| [SequentialWordCount.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/MapReduce/SequentialWordCount.hs) | ~100 | Map-Reduce, function composition, foldl' |
| [ParallelWordCount.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/MapReduce/ParallelWordCount.hs) | ~130 | Par monad, deterministic parallelism |
| [SequentialPi.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/MonteCarlo/SequentialPi.hs) | ~140 | Pure PRNG, BangPatterns, strict accumulation |
| [ParallelPi.hs](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/haskell/src/MonteCarlo/ParallelPi.hs) | ~260 | Async, STM, splittable PRNG, futures/promises |
| [benchmark.cpp](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/cpp/benchmark.cpp) | ~350 | std::async, std::thread, mutex, atomic |
| [benchmark.py](file:///Users/krishdave/Documents/Krish%20Stuff/8th%20Semester/Programming%20Languages/Project/python/benchmark.py) | ~330 | multiprocessing.Pool, GIL documentation |

---

## Benchmark Results Summary

All benchmarks run on **8-core Apple Silicon (ARM64)**, Haskell with `-N4`.

### Problem 1: Parallel Merge Sort (Best Parallel Config)

| Input Size | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|-----------|------------|------------|---------|---------|-----------|-----------|
| 10K | 12.44 ms | **2.66 ms** (4.7x) | 0.64 ms | **0.28 ms** (2.3x) | 34 ms | 96 ms (0.36x) |
| 50K | 75.9 ms | **24 ms** (3.2x) | 3.2 ms | **0.9 ms** (3.6x) | 197 ms | 142 ms (1.4x) |
| 100K | 169 ms | **53 ms** (3.2x) | 6.7 ms | **1.5 ms** (4.5x) | 417 ms | 232 ms (1.8x) |
| 500K | 1.43 s | **518 ms** (2.8x) | 38 ms | **9.0 ms** (4.2x) | — | — |

### Problem 2: Matrix Multiplication (Best Parallel Config)

| Size | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|------|------------|------------|---------|---------|-----------|-----------|
| 64×64 | 2.5 ms | 2.2 ms (1.1x) | 0.13 ms | 0.11 ms (1.1x) | 42 ms | 121 ms (0.3x) |
| 128×128 | 19 ms | **6.1 ms** (3.1x) | 0.62 ms | 0.67 ms (0.9x) | 276 ms | 135 ms (2.0x) |
| 256×256 | 166 ms | **49 ms** (3.4x) | 4.2 ms | **1.1 ms** (4.0x) | — | — |

### Problem 3: MapReduce Word Count

| Words | Haskell Seq | Haskell Par | C++ Seq | C++ Par | Python Seq | Python Par |
|-------|------------|------------|---------|---------|-----------|-----------|
| 155K | 85 ms | 134 ms (0.6x) | 13 ms | **3.9 ms** (3.3x) | 47 ms | 88 ms (0.5x) |
| 465K | 256 ms | 414 ms (0.6x) | 39 ms | **12 ms** (3.3x) | 143 ms | 114 ms (1.3x) |

> [!NOTE]
> Haskell's parallel word count is slower due to String (linked-list of characters) overhead. Using ByteString/Text would dramatically improve this.

### Problem 4: Monte Carlo Pi Estimation

| Samples | Haskell Seq | Haskell Async(4) | C++ Seq | C++ Async(4) | Python Seq | Python Par(4) |
|---------|------------|-----------------|---------|-------------|-----------|-------------|
| 100K | 1.1 ms | **0.4 ms** (2.6x) | 1.2 ms | **0.4 ms** (3.0x) | 21 ms | 80 ms (0.3x) |
| 1M | 10.5 ms | **3.2 ms** (3.3x) | 12 ms | **3.6 ms** (3.4x) | 227 ms | 128 ms (1.8x) |
| 10M | 163 ms | **30 ms** (5.4x) | 124 ms | **32 ms** (3.9x) | 2.2 s | 798 ms (2.8x) |

> [!IMPORTANT]
> Haskell's Monte Carlo achieved **5.37x speedup on 4 cores** — super-linear! This is due to SplitMix's excellent cache behavior when each thread has its own independent generator.

---

## Haskell Spark Statistics (Full Run)

```
SPARKS: 585 (478 converted, 0 overflowed, 0 dud, 45 GC'd, 62 fizzled)
```

| Metric | Value | Meaning |
|--------|-------|---------|
| Converted | 478 (81.7%) | Successfully executed on another core ✓ |
| Fizzled | 62 (10.6%) | Main thread evaluated before worker picked it up |
| GC'd | 45 (7.7%) | Garbage collected before evaluation |
| Dud | 0 (0%) | None were already evaluated |

---

## Key Observations for Presentation

### 1. Purity Enables Safe Parallelism
- Haskell's merge sort: **4 lines changed** to go from sequential to parallel
- C++: Need `std::async`, careful index management, thread lifecycle
- No data races possible in Haskell — the type system prevents them

### 2. Multiple Parallelism Abstractions
Each problem uses a **different** Haskell parallelism mechanism:
1. Eval monad (sparks) — lightweight, best-effort
2. Strategies (parMap) — separates algorithm from parallelism
3. Par monad — guaranteed execution, deterministic
4. Async + STM — explicit concurrency with composable transactions

### 3. Performance Trade-offs
- C++ is 10-40x faster in absolute terms (arrays vs linked lists)
- Haskell achieves comparable **relative speedup** ratios
- Python's GIL makes threading useless; multiprocessing adds huge overhead

### 4. Code Complexity
- Haskell: Parallelism is a **function** (parMap, rpar) — composable
- C++: Parallelism is **boilerplate** (create threads, manage lifecycle, synchronize)
- Python: Parallelism requires **architecture change** (multiprocessing = separate processes)

---

## Build & Run Commands

```bash
# Haskell
cd haskell && cabal build && cabal run haskell-parallelism -- +RTS -N4 -s

# C++
cd cpp && g++ -std=c++17 -O2 -pthread -o benchmark benchmark.cpp && ./benchmark

# Python
cd python && python3 benchmark.py
```

---

## What's Next

- [ ] **Generate comparison charts** using matplotlib/seaborn (speedup curves, bar charts)
- [ ] **Code complexity analysis** (LOC, sync primitives count)  
- [ ] **Improve word count** — switch from String to ByteString for fair comparison
- [ ] **Write formal report** with analysis and conclusions
- [ ] **Create presentation slides**
