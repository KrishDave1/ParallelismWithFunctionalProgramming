/*
================================================================================
  C++ Imperative Comparison: Parallel Algorithms
================================================================================

PURPOSE:
  This file implements the same four computational problems in C++ to serve
  as an IMPERATIVE BASELINE for comparison with our Haskell implementations.

  The key differences to observe are:
  1. SHARED MUTABLE STATE: C++ threads share memory, requiring explicit
     synchronization (mutexes, atomics) to prevent data races.
  2. MANUAL THREAD MANAGEMENT: We must create, join, and manage threads.
  3. IN-PLACE MUTATION: C++ sorts and modifies arrays in place.
  4. EXPLICIT SYNCHRONIZATION: Locks, condition variables, atomics.

  These are the "pain points" that FP's purity eliminates.

COMPILE:
  g++ -std=c++17 -O2 -pthread -o benchmark benchmark.cpp

RUN:
  ./benchmark
================================================================================
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <future>
#include <random>
#include <mutex>
#include <map>
#include <string>
#include <sstream>
#include <numeric>
#include <functional>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <atomic>
#include <cctype>

using namespace std;
using namespace chrono;

// ============================================================================
// Timing utility
// ============================================================================

template <typename Func>
double timeIt(Func f) {
    auto start = high_resolution_clock::now();
    f();
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0; // ms
}

// ============================================================================
// PROBLEM 1: Merge Sort
// ============================================================================

/*
  IMPERATIVE MERGE SORT (Sequential)
  
  KEY DIFFERENCES FROM HASKELL:
  - IN-PLACE: We modify the array directly (no new allocations per level)
  - MUTABLE: The 'temp' buffer is reused across merge operations
  - INDEX-BASED: We use array indices, not recursive list splitting
  
  This is more memory-efficient than the Haskell version but:
  - Harder to reason about correctness (aliasing, buffer overflows)
  - Cannot safely share arrays between threads without synchronization
  - Must carefully manage the temp buffer's lifecycle
*/
void merge(vector<int>& arr, int left, int mid, int right, vector<int>& temp) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    for (int i = left; i <= right; i++) arr[i] = temp[i];
}

void sequentialMergeSort(vector<int>& arr, int left, int right, vector<int>& temp) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    sequentialMergeSort(arr, left, mid, temp);
    sequentialMergeSort(arr, mid + 1, right, temp);
    merge(arr, left, mid, right, temp);
}

/*
  IMPERATIVE PARALLEL MERGE SORT
  
  KEY DIFFERENCES FROM HASKELL:
  - Uses std::async to spawn threads (similar to Haskell's 'async')
  - SHARED MUTABLE STATE: Both threads write to the SAME array
    (safe here because they write to different index ranges)
  - Must explicitly manage thread lifecycle (future.get() = join)
  - Depth threshold to avoid thread explosion (same concept as Haskell)
  
  DANGER ZONES:
  - If we mess up index ranges, we get data races → undefined behavior
  - No type-system protection against shared mutable access
  - Must remember to call .get() on futures (else threads are detached)
  
  CONTRAST WITH HASKELL:
  - Haskell's rpar creates lightweight sparks, not OS threads
  - Haskell's purity guarantees no data races AT COMPILE TIME
  - C++ requires the programmer to manually verify thread safety
*/
void parallelMergeSort(vector<int>& arr, int left, int right, vector<int>& temp, int depth) {
    if (left >= right) return;
    if (depth <= 0) {
        sequentialMergeSort(arr, left, right, temp);
        return;
    }
    int mid = left + (right - left) / 2;
    
    // Spawn left half on a new thread
    // NOTE: This shares 'arr' and 'temp' with the parent thread!
    // Safe ONLY because we write to non-overlapping index ranges.
    auto leftFuture = async(launch::async, [&]() {
        parallelMergeSort(arr, left, mid, temp, depth - 1);
    });
    
    // Right half on current thread
    parallelMergeSort(arr, mid + 1, right, temp, depth - 1);
    
    // Wait for left half to complete
    leftFuture.get();
    
    // Merge (sequential — both halves are now sorted)
    merge(arr, left, mid, right, temp);
}

// ============================================================================
// PROBLEM 2: Matrix Multiplication
// ============================================================================

/*
  IMPERATIVE MATRIX MULTIPLICATION (Sequential)
  
  Three nested for-loops — the classic O(n³) algorithm.
  
  KEY DIFFERENCES FROM HASKELL:
  - IN-PLACE: Result matrix is pre-allocated and filled
  - INDEX-BASED: No list comprehensions, just index arithmetic
  - CACHE-FRIENDLY: Row-major access pattern is efficient
  
  The C++ version is MUCH faster than Haskell's list-based version
  because arrays have better cache locality than linked lists.
*/
vector<vector<double>> sequentialMatMul(const vector<vector<double>>& a,
                                        const vector<vector<double>>& b) {
    int n = a.size();
    vector<vector<double>> c(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                c[i][j] += a[i][k] * b[k][j];
    return c;
}

/*
  IMPERATIVE PARALLEL MATRIX MULTIPLICATION
  
  Uses std::thread to parallelize the outer loop.
  Each thread computes a chunk of rows.
  
  KEY DIFFERENCES FROM HASKELL:
  - Haskell: parMap rdeepseq computeRow a (one line change!)
  - C++: Must manually partition rows, create threads, join them
  - C++: Threads share the output matrix (safe: non-overlapping writes)
  - C++: Must carefully compute chunk boundaries
*/
vector<vector<double>> parallelMatMul(const vector<vector<double>>& a,
                                       const vector<vector<double>>& b,
                                       int numThreads) {
    int n = a.size();
    vector<vector<double>> c(n, vector<double>(n, 0.0));
    
    vector<thread> threads;
    int chunkSize = (n + numThreads - 1) / numThreads;
    
    for (int t = 0; t < numThreads; t++) {
        int startRow = t * chunkSize;
        int endRow = min(startRow + chunkSize, n);
        
        threads.emplace_back([&, startRow, endRow]() {
            for (int i = startRow; i < endRow; i++)
                for (int k = 0; k < n; k++)
                    for (int j = 0; j < n; j++)
                        c[i][j] += a[i][k] * b[k][j];
        });
    }
    
    for (auto& t : threads) t.join();
    return c;
}

// ============================================================================
// Random Data Generation
// ============================================================================

vector<int> generateRandomList(int n, int seed) {
    mt19937 gen(seed);
    uniform_int_distribution<int> dist(1, n * 10);
    vector<int> v(n);
    for (int i = 0; i < n; i++) v[i] = dist(gen);
    return v;
}

vector<vector<double>> generateRandomMatrix(int n, int seed) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(0.0, 100.0);
    vector<vector<double>> m(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = dist(gen);
    return m;
}


// ============================================================================
// Helper: Print formatted result
// ============================================================================

void printResult(const string& label, double timeMs) {
    cout << "  " << left << setw(40) << label;
    if (timeMs < 1.0) cout << fixed << setprecision(0) << timeMs * 1000 << "us" << endl;
    else if (timeMs < 1000.0) cout << fixed << setprecision(2) << timeMs << " ms" << endl;
    else cout << fixed << setprecision(3) << timeMs / 1000.0 << " s" << endl;
}

// ============================================================================
// MAIN: Run all benchmarks
// ============================================================================

int main() {
    cout << endl;
    cout << "╔══════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║   C++ Imperative Comparison Benchmarks                         ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════════╝" << endl;
    
    // ===== PROBLEM 1: Merge Sort =====
    cout << "\n======================================================================" << endl;
    cout << "  BENCHMARK 1: Merge Sort (std::async / std::thread)" << endl;
    cout << "======================================================================\n" << endl;
    
    for (int size : {10000, 50000, 100000, 500000}) {
        cout << "  --- Input size: " << size << " elements ---" << endl;
        auto original = generateRandomList(size, 42);
        
        { // Sequential
            auto arr = original;
            vector<int> temp(size);
            double t = timeIt([&]() {
                sequentialMergeSort(arr, 0, size - 1, temp);
            });
            printResult("Sequential Merge Sort", t);
            
            // Parallel
            for (int depth : {2, 3, 4}) {
                auto arr2 = original;
                vector<int> temp2(size);
                double pt = timeIt([&]() {
                    parallelMergeSort(arr2, 0, size - 1, temp2, depth);
                });
                string label = "Parallel (depth=" + to_string(depth) + ")";
                printResult(label, pt);
                cout << "    Speedup: " << fixed << setprecision(2) << t / pt << "x" << endl;
                
                // Verify correctness
                bool correct = (arr == arr2);
                cout << "    Correctness: " << (correct ? "PASS" : "FAIL") << endl;
            }
        }
        cout << endl;
    }
    
    // ===== PROBLEM 2: Matrix Multiplication =====
    cout << "======================================================================" << endl;
    cout << "  BENCHMARK 2: Matrix Multiplication (std::thread)" << endl;
    cout << "======================================================================\n" << endl;
    
    for (int size : {64, 128, 256}) {
        cout << "  --- Matrix size: " << size << "x" << size << " ---" << endl;
        auto a = generateRandomMatrix(size, 42);
        auto b = generateRandomMatrix(size, 99);
        
        double seqTime;
        {
            seqTime = timeIt([&]() { sequentialMatMul(a, b); });
            printResult("Sequential", seqTime);
        }
        
        for (int threads : {2, 4, 8}) {
            double pt = timeIt([&]() { parallelMatMul(a, b, threads); });
            string label = "Parallel (" + to_string(threads) + " threads)";
            printResult(label, pt);
            cout << "    Speedup: " << fixed << setprecision(2) << seqTime / pt << "x" << endl;
        }
        cout << endl;
    }

    
    cout << "\nAll C++ benchmarks complete!" << endl;
    return 0;
}
