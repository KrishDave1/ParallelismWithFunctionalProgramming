# Task: Harnessing Functional Programming for Parallelism (Plan A - Haskell)

## Phase 1: Project Setup & Foundation
- [x] Set up Haskell project structure with cabal
- [x] Set up C++ imperative comparison project
- [x] Set up Python imperative comparison project
- [x] Create benchmarking harness (input gen, timing, CSV output)
- [x] Write comprehensive README with project overview

## Phase 2: Problem 1 — Parallel Merge Sort
- [x] Haskell: Sequential merge sort
- [x] Haskell: Parallel merge sort (Eval monad, rpar/rseq, depth threshold)
- [x] C++: Sequential + parallel merge sort (std::async)
- [x] Python: Sequential + parallel merge sort (multiprocessing)
- [x] Benchmark across sizes (10K, 50K, 100K, 500K) and cores (1,2,4,8)
- [x] Document results

## Phase 3: Problem 2 — Parallel Matrix Multiplication
- [x] Haskell: Sequential + parallel matrix multiply (parMap + chunking)
- [x] C++: Sequential + parallel (std::thread)
- [x] Python: Sequential + parallel (multiprocessing)
- [x] Benchmark across sizes (64, 128, 256)
- [x] Document results

## Phase 4: Problem 3 — Parallel MapReduce Word Count
- [x] Haskell: Sequential + parallel (Par monad, parMap)
- [x] C++: Sequential + parallel (std::async)
- [x] Python: Sequential + parallel (multiprocessing)
- [x] Use generated text corpus for testing
- [x] Benchmark and document results

## Phase 5: Problem 4 — Monte Carlo Pi Estimation
- [x] Haskell: Sequential + parallel (async, STM)
- [x] C++: Sequential + parallel (std::async/std::future, atomic)
- [x] Python: Sequential + parallel (multiprocessing)
- [x] Benchmark across sample sizes (100K, 1M, 10M)
- [x] Document results

## Phase 6: Problem 5 — K-Means Clustering
- [x] Haskell: Sequential + parallel (Strategies, parListChunk)
- [x] C++: Sequential + parallel (std::async)
- [x] Python: Sequential + parallel (multiprocessing)
- [x] Benchmark across point counts (10K, 50K, 100K)
- [x] Document results

## Phase 7: Problem 6 — Numerical Integration
- [x] Haskell: Sequential + parallel (Domain decomposition)
- [x] C++: Sequential + parallel (std::async)
- [x] Python: Sequential + parallel (multiprocessing)
- [x] Benchmark across intervals (1M, 10M, 50M)
- [x] Document results

## Phase 8: Analysis & Presentation
- [ ] Generate all benchmark charts (matplotlib/seaborn)
- [ ] Code complexity comparison (LOC, sync primitives)
- [ ] Write comprehensive project report
- [ ] Create presentation slides
