"""
================================================================================
  Python Imperative Comparison: Parallel Algorithms
================================================================================

PURPOSE:
  This file implements the same four computational problems in Python to serve
  as a second IMPERATIVE BASELINE for comparison with Haskell implementations.

  Python adds an interesting dimension to the comparison:
  - GIL (Global Interpreter Lock) prevents true multi-threading for CPU-bound work
  - Must use multiprocessing (separate processes) for CPU parallelism
  - This means inter-process communication overhead (pickling/unpickling)
  - Shows WHY language-level parallelism support matters

KEY OBSERVATIONS:
  1. Python's GIL makes threading useless for CPU-bound parallelism
  2. multiprocessing.Pool adds significant overhead (process creation, IPC)
  3. Python is 10-100x slower than Haskell/C++ for CPU-bound work
  4. BUT: Python code is very readable — good for showing what we're computing

RUN:
  python3 benchmark.py

================================================================================
"""

import time
import random
import math
import sys
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import reduce
from collections import Counter

# ============================================================================
# Timing utility
# ============================================================================

def time_it(func, *args):
    """
    Measure wall-clock time of a function call.
    Returns (result, elapsed_ms).
    """
    start = time.perf_counter()
    result = func(*args)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    return result, elapsed


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def print_result(label, time_ms):
    if time_ms < 1.0:
        print(f"  {label:<40}{time_ms*1000:.0f} μs")
    elif time_ms < 1000:
        print(f"  {label:<40}{time_ms:.2f} ms")
    else:
        print(f"  {label:<40}{time_ms/1000:.3f} s")


# ============================================================================
# PROBLEM 1: Merge Sort
# ============================================================================

def merge(left, right):
    """
    Merge two sorted lists into one sorted list.
    
    COMPARISON WITH HASKELL:
      Haskell: merge [] ys = ys; merge xs [] = xs; merge (x:xs) (y:ys) = ...
      Python: Imperative with explicit indices and list building
      
    Both create a new list (neither is truly in-place).
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def sequential_merge_sort(arr):
    """
    Sequential merge sort.
    
    COMPARISON WITH HASKELL:
      Very similar structure! Python's slice notation makes it almost as clean:
        Haskell: let (left, right) = splitAt mid xs
        Python:  left, right = arr[:mid], arr[mid:]
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid])
    right = sequential_merge_sort(arr[mid:])
    return merge(left, right)


def _sort_chunk(arr):
    """Worker function for parallel sort (must be top-level for pickling)."""
    return sequential_merge_sort(arr)


def parallel_merge_sort(arr, num_workers=4):
    """
    Parallel merge sort using multiprocessing.
    
    CRITICAL DIFFERENCE FROM HASKELL:
      Python's GIL prevents true thread-level parallelism.
      We MUST use multiprocessing, which means:
      1. Each worker is a SEPARATE PROCESS (heavy to create)
      2. Data must be SERIALIZED (pickled) to send to workers
      3. Results must be DESERIALIZED when received back
      4. This IPC overhead can dominate for small inputs
      
      Haskell:
        - Sparks are nearly free (just a pointer on a work queue)
        - Data sharing via immutable memory (no copying needed)
        - GC handles cleanup automatically
    
    STRATEGY:
      1. Split array into num_workers chunks
      2. Sort each chunk in a separate process
      3. Merge sorted chunks pairwise
    """
    if len(arr) <= 1000:  # Don't parallelize small arrays
        return sequential_merge_sort(arr)
    
    chunk_size = max(1, len(arr) // num_workers)
    chunks = [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]
    
    with Pool(num_workers) as pool:
        sorted_chunks = pool.map(_sort_chunk, chunks)
    
    # Merge sorted chunks pairwise
    while len(sorted_chunks) > 1:
        merged = []
        for i in range(0, len(sorted_chunks), 2):
            if i + 1 < len(sorted_chunks):
                merged.append(merge(sorted_chunks[i], sorted_chunks[i+1]))
            else:
                merged.append(sorted_chunks[i])
        sorted_chunks = merged
    
    return sorted_chunks[0]


# ============================================================================
# PROBLEM 2: Matrix Multiplication
# ============================================================================

def sequential_mat_mul(a, b):
    """
    Sequential matrix multiplication using nested loops.
    
    COMPARISON WITH HASKELL:
      Haskell: [[ sum $ zipWith (*) row col | col <- transpose b ] | row <- a]
      Python:  Three nested for-loops (C-style iteration)
      
    The Haskell version is more declarative and closer to mathematical notation.
    """
    n = len(a)
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c


def _compute_rows(args):
    """Worker: compute a chunk of result rows."""
    a_chunk, b, n = args
    result = []
    for row in a_chunk:
        new_row = [0.0] * n
        for k in range(n):
            for j in range(n):
                new_row[j] += row[k] * b[k][j]
        result.append(new_row)
    return result


def parallel_mat_mul(a, b, num_workers=4):
    """
    Parallel matrix multiplication using multiprocessing.
    
    Each worker computes a chunk of rows.
    
    COMPARISON WITH HASKELL:
      Haskell: parMap rdeepseq computeRow a  ← one line!
      Python:  Split data, pickle, send to processes, unpickle, collect, merge
    """
    n = len(a)
    chunk_size = max(1, n // num_workers)
    chunks = [(a[i:i+chunk_size], b, n) for i in range(0, n, chunk_size)]
    
    with Pool(num_workers) as pool:
        results = pool.map(_compute_rows, chunks)
    
    return [row for chunk in results for row in chunk]


# ============================================================================
# PROBLEM 3: Word Count (MapReduce)
# ============================================================================

def sequential_word_count(text):
    """
    Sequential word count using Counter.
    
    Python's Counter makes this very clean — arguably as clean as Haskell:
      Haskell: foldl' (\acc w -> Map.insertWith (+) w 1 acc) Map.empty words
      Python:  Counter(words)
    
    But Python's Counter uses mutable hash tables internally.
    """
    words = text.lower().split()
    words = [''.join(c for c in w if c.isalpha()) for w in words]
    words = [w for w in words if w]
    return Counter(words)


def _count_chunk(text):
    """Worker: count words in a text chunk."""
    words = text.lower().split()
    words = [''.join(c for c in w if c.isalpha()) for w in words]
    words = [w for w in words if w]
    return Counter(words)


def parallel_word_count(text, num_workers=4):
    """
    Parallel word count using multiprocessing.
    
    COMPARISON WITH HASKELL:
      Haskell: runPar $ parMapM countChunk chunks >>= return . unionsWith (+)
      Python:  Pool.map(count_chunk, chunks); reduce Counter merge
    
    OVERHEAD: The text must be pickled and sent to each worker process.
    For large texts, this serialization dominates.
    """
    chunk_size = max(1, len(text) // num_workers)
    # Split on word boundaries
    chunks = []
    start = 0
    for i in range(1, num_workers):
        end = min(i * chunk_size, len(text))
        # Don't split in the middle of a word
        while end < len(text) and text[end] != ' ':
            end += 1
        chunks.append(text[start:end])
        start = end
    chunks.append(text[start:])
    
    with Pool(num_workers) as pool:
        results = pool.map(_count_chunk, chunks)
    
    # Reduce: merge all counters
    total = Counter()
    for result in results:
        total += result
    return total


# ============================================================================
# PROBLEM 4: Monte Carlo Pi Estimation
# ============================================================================

def sequential_monte_carlo_pi(n, seed=42):
    """
    Sequential Monte Carlo Pi estimation.
    
    COMPARISON WITH HASKELL:
      Haskell: Uses pure SplitMix PRNG (no mutable state)
      Python:  Uses random module (mutable internal state)
      
    Python's random module is NOT thread-safe — must use separate
    Random instances in each process (similar problem as C++).
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            hits += 1
    return 4.0 * hits / n


def _monte_carlo_worker(args):
    """Worker: count hits in a chunk of samples."""
    n, seed = args
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            hits += 1
    return hits


def parallel_monte_carlo_pi(n, num_workers=4, seed=42):
    """
    Parallel Monte Carlo Pi estimation using multiprocessing.
    
    COMPARISON WITH HASKELL:
      Haskell:
        - Split PRNG purely: let (gen1, gen2) = splitSMGen gen
        - Use mapConcurrently (futures) or STM (transactional memory)
        - Deterministic: same seed → same split → same result
      
      Python:
        - Must create separate Random instances per process
        - No pure 'split' operation → use different seeds
        - IPC overhead for result collection
    """
    samples_per_worker = n // num_workers
    tasks = [(samples_per_worker, seed + i * 1000) for i in range(num_workers)]
    
    with Pool(num_workers) as pool:
        results = pool.map(_monte_carlo_worker, tasks)
    
    total_hits = sum(results)
    return 4.0 * total_hits / (num_workers * samples_per_worker)


# ============================================================================
# PROBLEM 5: K-Means Clustering (Machine Learning)
# ============================================================================

def _distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def _assign_point(centroids, point):
    dists = [_distance(point, c) for c in centroids]
    return dists.index(min(dists))

def sequential_kmeans(points, init_centroids, k, max_iter, epsilon=0.001):
    centroids = list(init_centroids)
    for _ in range(max_iter):
        # Assign
        assignments = [_assign_point(centroids, p) for p in points]
        # Update
        new_centroids = []
        converged = True
        for i in range(k):
            cluster = [points[j] for j in range(len(points)) if assignments[j] == i]
            if not cluster:
                new_centroids.append(centroids[i])
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            new_c = (cx, cy)
            if _distance(centroids[i], new_c) > epsilon:
                converged = False
            new_centroids.append(new_c)
        centroids = new_centroids
        if converged:
            break
    return centroids

def _assign_chunk(args):
    """Worker: assign a chunk of points to centroids."""
    chunk, centroids = args
    return [_assign_point(centroids, p) for p in chunk]

def parallel_kmeans(points, init_centroids, k, max_iter, num_workers=4, epsilon=0.001):
    centroids = list(init_centroids)
    chunk_size = max(1, len(points) // num_workers)
    
    for _ in range(max_iter):
        # Parallel assign
        chunks = [(points[i:i+chunk_size], centroids)
                  for i in range(0, len(points), chunk_size)]
        
        with Pool(num_workers) as pool:
            results = pool.map(_assign_chunk, chunks)
        
        assignments = [a for chunk_result in results for a in chunk_result]
        
        # Update
        new_centroids = []
        converged = True
        for i in range(k):
            cluster = [points[j] for j in range(len(points)) if assignments[j] == i]
            if not cluster:
                new_centroids.append(centroids[i])
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            new_c = (cx, cy)
            if _distance(centroids[i], new_c) > epsilon:
                converged = False
            new_centroids.append(new_c)
        centroids = new_centroids
        if converged:
            break
    return centroids

def generate_clustered_data(k, total_points, seed=42):
    points = []
    per_cluster = total_points // k
    for c in range(k):
        cx = 100.0 * math.cos(2 * math.pi * c / k)
        cy = 100.0 * math.sin(2 * math.pi * c / k)
        rng = random.Random(seed + c)
        for _ in range(per_cluster):
            points.append((cx + rng.uniform(-20, 20), cy + rng.uniform(-20, 20)))
    return points


# ============================================================================
# PROBLEM 6: Numerical Integration (Numerical Simulation)
# ============================================================================

# Define test functions at module level for pickling
def _func_sin(x): return math.sin(x)
def _func_leibniz_pi(x): return 4.0 / (1.0 + x * x)
def _func_x_squared(x): return x * x

_FUNC_MAP = {
    'sin': _func_sin,
    'leibniz': _func_leibniz_pi,
    'x2': _func_x_squared,
}

def sequential_integrate(f, a, b, n):
    """Trapezoidal rule integration."""
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + h * i)
    return h * total

def _integrate_chunk(args):
    """Worker: integrate f over sub-range [lo, hi]."""
    func_name, lo, hi, sub_n = args
    f = _FUNC_MAP[func_name]
    h = (hi - lo) / sub_n
    total = 0.5 * (f(lo) + f(hi))
    for i in range(1, sub_n):
        total += f(lo + h * i)
    return h * total

def parallel_integrate(func_name, a, b, n, num_workers=4):
    """Parallel integration via domain decomposition."""
    sub_n = n // num_workers
    h = (b - a) / num_workers
    tasks = [(func_name, a + h * t, a + h * (t + 1), sub_n) for t in range(num_workers)]
    
    with Pool(num_workers) as pool:
        results = pool.map(_integrate_chunk, tasks)
    
    return sum(results)


# ============================================================================
# Data Generation
# ============================================================================

def generate_random_list(n, seed=42):
    rng = random.Random(seed)
    return [rng.randint(1, n * 10) for _ in range(n)]


def generate_random_matrix(n, seed=42):
    rng = random.Random(seed)
    return [[rng.uniform(0, 100) for _ in range(n)] for _ in range(n)]


def generate_text(num_words, seed=42):
    word_list = [
        "the", "quick", "brown", "fox", "jumps", "over",
        "functional", "programming", "parallelism",
        "haskell", "purity", "immutability"
    ]
    rng = random.Random(seed)
    return " ".join(rng.choice(word_list) for _ in range(num_words))


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Python Imperative Comparison Benchmarks                      ║")
    print("║   NOTE: Uses multiprocessing (not threading) due to GIL        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # ===== PROBLEM 1: Merge Sort =====
    print_header("BENCHMARK 1: Merge Sort (multiprocessing.Pool)")
    
    for size in [10000, 50000, 100000]:
        print(f"  --- Input size: {size} elements ---")
        arr = generate_random_list(size)
        
        seq_result, seq_time = time_it(sequential_merge_sort, arr)
        print_result("Sequential Merge Sort", seq_time)
        
        for workers in [2, 4]:
            par_result, par_time = time_it(parallel_merge_sort, arr, workers)
            speedup = seq_time / par_time if par_time > 0 else 0
            print_result(f"Parallel ({workers} workers)", par_time)
            print(f"    Speedup: {speedup:.2f}x")
            correct = seq_result == par_result
            print(f"    Correctness: {'✓ PASS' if correct else '✗ FAIL'}")
        print()
    
    # ===== PROBLEM 2: Matrix Multiplication =====
    print_header("BENCHMARK 2: Matrix Multiplication (multiprocessing.Pool)")
    
    for size in [64, 128]:
        print(f"  --- Matrix size: {size}x{size} ---")
        a = generate_random_matrix(size, 42)
        b = generate_random_matrix(size, 99)
        
        _, seq_time = time_it(sequential_mat_mul, a, b)
        print_result("Sequential", seq_time)
        
        for workers in [2, 4]:
            _, par_time = time_it(parallel_mat_mul, a, b, workers)
            speedup = seq_time / par_time if par_time > 0 else 0
            print_result(f"Parallel ({workers} workers)", par_time)
            print(f"    Speedup: {speedup:.2f}x")
        print()
    
    # ===== PROBLEM 3: Word Count =====
    print_header("BENCHMARK 3: Word Count (multiprocessing.Pool)")
    
    for num_words in [50000, 150000]:
        print(f"  --- Text size: ~{num_words} words ---")
        text = generate_text(num_words)
        
        seq_result, seq_time = time_it(sequential_word_count, text)
        print_result("Sequential", seq_time)
        print(f"    Top 5: {seq_result.most_common(5)}")
        
        for workers in [2, 4]:
            par_result, par_time = time_it(parallel_word_count, text, workers)
            speedup = seq_time / par_time if par_time > 0 else 0
            print_result(f"Parallel ({workers} workers)", par_time)
            print(f"    Speedup: {speedup:.2f}x")
        print()
    
    # ===== PROBLEM 4: Monte Carlo Pi =====
    print_header("BENCHMARK 4: Monte Carlo Pi (multiprocessing.Pool)")
    
    for n in [100000, 1000000, 10000000]:
        print(f"  --- Samples: {n} ---")
        
        seq_result, seq_time = time_it(sequential_monte_carlo_pi, n)
        print_result("Sequential", seq_time)
        print(f"    π ≈ {seq_result:.7f}")
        
        for workers in [2, 4, 8]:
            par_result, par_time = time_it(parallel_monte_carlo_pi, n, workers)
            speedup = seq_time / par_time if par_time > 0 else 0
            print_result(f"Parallel ({workers} workers)", par_time)
            print(f"    π ≈ {par_result:.7f}")
            print(f"    Speedup: {speedup:.2f}x")
        print()
    
    # ===== PROBLEM 5: K-Means Clustering (Machine Learning) =====
    print_header("BENCHMARK 5: K-Means Clustering (multiprocessing.Pool)")
    
    for num_points in [10000, 50000, 100000]:
        print(f"  --- {num_points} points, K=5 ---")
        points = generate_clustered_data(5, num_points)
        init_centroids = points[:5]
        
        seq_result, seq_time = time_it(sequential_kmeans, points, init_centroids, 5, 100)
        print_result("Sequential K-Means", seq_time)
        
        for workers in [2, 4]:
            par_result, par_time = time_it(parallel_kmeans, points, init_centroids, 5, 100, workers)
            speedup = seq_time / par_time if par_time > 0 else 0
            print_result(f"Parallel ({workers} workers)", par_time)
            print(f"    Speedup: {speedup:.2f}x")
        print()
    
    # ===== PROBLEM 6: Numerical Integration (Numerical Simulation) =====
    print_header("BENCHMARK 6: Numerical Integration — Trapezoidal Rule")
    
    test_funcs = [
        ("sin(x) on [0,π]", _func_sin, 'sin', 0, math.pi, 2.0),
        ("4/(1+x²) on [0,1] = π", _func_leibniz_pi, 'leibniz', 0, 1, math.pi),
        ("x² on [0,1] = 1/3", _func_x_squared, 'x2', 0, 1, 1.0/3.0),
    ]
    
    for name, f, func_name, a, b, exact in test_funcs:
        print(f"  === {name} ===")
        for n in [1000000, 10000000]:
            print(f"  --- N = {n} ---")
            
            seq_result, seq_time = time_it(sequential_integrate, f, a, b, n)
            print_result("Sequential", seq_time)
            print(f"    Result: {seq_result:.15f}")
            
            for workers in [2, 4, 8]:
                par_result, par_time = time_it(parallel_integrate, func_name, a, b, n, workers)
                speedup = seq_time / par_time if par_time > 0 else 0
                print_result(f"Parallel ({workers} workers)", par_time)
                print(f"    Speedup: {speedup:.2f}x")
            print()
    
    print("\nAll Python benchmarks complete!")


if __name__ == "__main__":
    main()
