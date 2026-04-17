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
// PROBLEM 3: Word Count (MapReduce)
// ============================================================================

/*
  IMPERATIVE WORD COUNT
  
  KEY DIFFERENCES FROM HASKELL:
  - Uses mutable unordered_map (hash map) for counting
  - Mutation-based: increment counts in place
  - String manipulation is manual (tolower, isalpha)
  
  PARALLEL VERSION:
  - Each thread counts words in its chunk → local map
  - Merge all local maps at the end
  - Must use mutex if threads share a map (or use thread-local maps)
  
  CONTRAST WITH HASKELL:
  - Haskell's Map.insertWith (+) is pure: creates new map each time
  - Haskell's tokenize is a pipeline of pure transformations
  - C++ modifies strings in place and imperatively builds the map
*/
map<string, int> sequentialWordCount(const string& text) {
    map<string, int> freq;
    string word;
    for (char c : text) {
        if (isalpha(c)) {
            word += tolower(c);
        } else if (!word.empty()) {
            freq[word]++;
            word.clear();
        }
    }
    if (!word.empty()) freq[word]++;
    return freq;
}

map<string, int> parallelWordCount(const string& text, int numThreads) {
    int chunkSize = text.size() / numThreads;
    vector<future<map<string, int>>> futures;
    
    for (int t = 0; t < numThreads; t++) {
        int start = t * chunkSize;
        int end = (t == numThreads - 1) ? text.size() : (t + 1) * chunkSize;
        
        // Adjust boundaries to not split words
        while (end < (int)text.size() && isalpha(text[end])) end++;
        
        futures.push_back(async(launch::async, [&text, start, end]() {
            map<string, int> local;
            string word;
            for (int i = start; i < end; i++) {
                char c = text[i];
                if (isalpha(c)) {
                    word += tolower(c);
                } else if (!word.empty()) {
                    local[word]++;
                    word.clear();
                }
            }
            if (!word.empty()) local[word]++;
            return local;
        }));
    }
    
    // Reduce: merge all local maps
    map<string, int> result;
    for (auto& f : futures) {
        auto local = f.get();
        for (auto& [word, count] : local) {
            result[word] += count;
        }
    }
    return result;
}

// ============================================================================
// PROBLEM 4: Monte Carlo Pi Estimation
// ============================================================================

/*
  IMPERATIVE MONTE CARLO PI
  
  KEY DIFFERENCES FROM HASKELL:
  - MUTABLE PRNG: std::mt19937 has mutable internal state
  - SHARED PRNG PROBLEM: Cannot safely share one PRNG across threads
    → Must create separate PRNGs per thread (using different seeds)
    → No elegant 'split' operation like Haskell's SplitMix
  
  PARALLEL VERSION:
  - Uses std::async with per-thread PRNGs (seeded by thread index)
  - Alternative: use atomic counter (shown in STM-equivalent version)
  
  CONTRAST WITH HASKELL:
  - Haskell: let (gen1, gen2) = split gen  ← pure, deterministic split
  - C++: mt19937 gen(seed + threadId)      ← ad-hoc, less rigorous
*/
double sequentialMonteCarloPi(int n, unsigned int seed) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    int hits = 0;
    for (int i = 0; i < n; i++) {
        double x = dist(gen);
        double y = dist(gen);
        if (x * x + y * y <= 1.0) hits++;
    }
    return 4.0 * hits / n;
}

double parallelMonteCarloPiAsync(int n, int numWorkers, unsigned int seed) {
    int samplesPerWorker = n / numWorkers;
    vector<future<int>> futures;
    
    for (int w = 0; w < numWorkers; w++) {
        futures.push_back(async(launch::async, [=]() {
            // Each thread gets its own PRNG with a different seed
            // NOTE: This is less rigorous than Haskell's splittable PRNG
            mt19937 gen(seed + w * 1000);
            uniform_real_distribution<double> dist(0.0, 1.0);
            int hits = 0;
            for (int i = 0; i < samplesPerWorker; i++) {
                double x = dist(gen);
                double y = dist(gen);
                if (x * x + y * y <= 1.0) hits++;
            }
            return hits;
        }));
    }
    
    int totalHits = 0;
    for (auto& f : futures) totalHits += f.get();
    return 4.0 * totalHits / (numWorkers * samplesPerWorker);
}

// Mutex-based version (equivalent to Haskell's STM version)
double parallelMonteCarloPiMutex(int n, int numWorkers, unsigned int seed) {
    int samplesPerWorker = n / numWorkers;
    atomic<int> totalHits{0};  // Atomic instead of mutex for simplicity
    
    vector<thread> threads;
    for (int w = 0; w < numWorkers; w++) {
        threads.emplace_back([&, w]() {
            mt19937 gen(seed + w * 1000);
            uniform_real_distribution<double> dist(0.0, 1.0);
            int localHits = 0;
            for (int i = 0; i < samplesPerWorker; i++) {
                double x = dist(gen);
                double y = dist(gen);
                if (x * x + y * y <= 1.0) localHits++;
            }
            totalHits += localHits;  // Atomic add
        });
    }
    
    for (auto& t : threads) t.join();
    return 4.0 * totalHits / (numWorkers * samplesPerWorker);
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

string generateText(int numWords) {
    vector<string> words = {
        "the", "quick", "brown", "fox", "jumps", "over",
        "functional", "programming", "parallelism",
        "haskell", "purity", "immutability"
    };
    mt19937 gen(42);
    uniform_int_distribution<int> dist(0, words.size() - 1);
    string text;
    for (int i = 0; i < numWords; i++) {
        if (i > 0) text += " ";
        text += words[dist(gen)];
    }
    return text;
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
    
    // ===== PROBLEM 3: Word Count =====
    cout << "======================================================================" << endl;
    cout << "  BENCHMARK 3: Word Count (std::async)" << endl;
    cout << "======================================================================\n" << endl;
    
    for (int numWords : {155000, 465000}) {
        cout << "  --- Text size: ~" << numWords << " words ---" << endl;
        string text = generateText(numWords);
        
        double seqTime = timeIt([&]() { sequentialWordCount(text); });
        printResult("Sequential", seqTime);
        
        for (int threads : {2, 4, 8}) {
            double pt = timeIt([&]() { parallelWordCount(text, threads); });
            string label = "Parallel (" + to_string(threads) + " threads)";
            printResult(label, pt);
            cout << "    Speedup: " << fixed << setprecision(2) << seqTime / pt << "x" << endl;
        }
        cout << endl;
    }
    
    // ===== PROBLEM 4: Monte Carlo Pi =====
    cout << "======================================================================" << endl;
    cout << "  BENCHMARK 4: Monte Carlo Pi (std::async + atomic)" << endl;
    cout << "======================================================================\n" << endl;
    
    for (int n : {100000, 1000000, 10000000}) {
        cout << "  --- Samples: " << n << " ---" << endl;
        
        double seqResult, seqTime;
        seqTime = timeIt([&]() { seqResult = sequentialMonteCarloPi(n, 42); });
        printResult("Sequential", seqTime);
        cout << "    π ≈ " << fixed << setprecision(7) << seqResult << endl;
        
        for (int workers : {2, 4, 8}) {
            double parResult;
            double pt = timeIt([&]() {
                parResult = parallelMonteCarloPiAsync(n, workers, 42);
            });
            string label = "Async (" + to_string(workers) + " workers)";
            printResult(label, pt);
            cout << "    π ≈ " << parResult << endl;
            cout << "    Speedup: " << fixed << setprecision(2) << seqTime / pt << "x" << endl;
        }
        
        {
            double mutexResult;
            double mt = timeIt([&]() {
                mutexResult = parallelMonteCarloPiMutex(n, 4, 42);
            });
            printResult("Atomic (4 workers)", mt);
            cout << "    π ≈ " << mutexResult << endl;
            cout << "    Speedup: " << fixed << setprecision(2) << seqTime / mt << "x" << endl;
        }
        cout << endl;
    }
    
    // ===== PROBLEM 5: K-Means Clustering (Machine Learning) =====
    cout << "======================================================================" << endl;
    cout << "  BENCHMARK 5: K-Means Clustering (std::async)" << endl;
    cout << "======================================================================\n" << endl;
    
    {
        struct Point { double x, y; };
        
        auto generateClusteredData = [](int k, int totalPoints, int seed) -> vector<Point> {
            vector<Point> points;
            int perCluster = totalPoints / k;
            for (int c = 0; c < k; c++) {
                double cx = 100.0 * cos(2.0 * M_PI * c / k);
                double cy = 100.0 * sin(2.0 * M_PI * c / k);
                mt19937 gen(seed + c);
                uniform_real_distribution<double> dist(-20.0, 20.0);
                for (int i = 0; i < perCluster; i++) {
                    points.push_back({cx + dist(gen), cy + dist(gen)});
                }
            }
            return points;
        };
        
        auto dist = [](const Point& a, const Point& b) {
            return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
        };
        
        auto assignPoint = [&dist](const vector<Point>& centroids, const Point& p) -> int {
            int best = 0;
            double bestDist = dist(p, centroids[0]);
            for (int i = 1; i < (int)centroids.size(); i++) {
                double d = dist(p, centroids[i]);
                if (d < bestDist) { bestDist = d; best = i; }
            }
            return best;
        };
        
        // Sequential K-Means
        auto seqKMeans = [&](const vector<Point>& points, vector<Point> centroids, int k, int maxIter) {
            for (int iter = 0; iter < maxIter; iter++) {
                // Assign
                vector<vector<Point>> clusters(k);
                for (const auto& p : points) {
                    int idx = assignPoint(centroids, p);
                    clusters[idx].push_back(p);  // MUTATION: accumulate in mutable vector
                }
                // Update
                vector<Point> newCentroids(k);
                bool converged = true;
                for (int i = 0; i < k; i++) {
                    if (clusters[i].empty()) { newCentroids[i] = centroids[i]; continue; }
                    double sx = 0, sy = 0;
                    for (const auto& p : clusters[i]) { sx += p.x; sy += p.y; }
                    newCentroids[i] = {sx / clusters[i].size(), sy / clusters[i].size()};
                    if (dist(centroids[i], newCentroids[i]) > 0.001) converged = false;
                }
                centroids = newCentroids;
                if (converged) break;
            }
            return centroids;
        };
        
        // Parallel K-Means
        auto parKMeans = [&](const vector<Point>& points, vector<Point> centroids, int k, int maxIter, int numThreads) {
            for (int iter = 0; iter < maxIter; iter++) {
                // Parallel assign
                int n = points.size();
                vector<int> assignments(n);
                vector<thread> threads;
                int chunkSize = (n + numThreads - 1) / numThreads;
                
                for (int t = 0; t < numThreads; t++) {
                    int start = t * chunkSize;
                    int end = min(start + chunkSize, n);
                    threads.emplace_back([&, start, end]() {
                        for (int i = start; i < end; i++)
                            assignments[i] = assignPoint(centroids, points[i]);
                    });
                }
                for (auto& t : threads) t.join();
                
                // Update centroids
                vector<double> sumX(k, 0), sumY(k, 0);
                vector<int> count(k, 0);
                for (int i = 0; i < n; i++) {
                    int idx = assignments[i];
                    sumX[idx] += points[i].x;
                    sumY[idx] += points[i].y;
                    count[idx]++;
                }
                
                bool converged = true;
                for (int i = 0; i < k; i++) {
                    if (count[i] == 0) continue;
                    Point newC = {sumX[i] / count[i], sumY[i] / count[i]};
                    if (dist(centroids[i], newC) > 0.001) converged = false;
                    centroids[i] = newC;
                }
                if (converged) break;
            }
            return centroids;
        };
        
        int k = 5;
        for (int numPoints : {10000, 50000, 100000, 200000}) {
            cout << "  --- " << numPoints << " points, K=" << k << " ---" << endl;
            auto points = generateClusteredData(k, numPoints, 42);
            vector<Point> initCentroids(points.begin(), points.begin() + k);
            
            double seqTime = timeIt([&]() { seqKMeans(points, initCentroids, k, 100); });
            printResult("Sequential K-Means", seqTime);
            
            for (int threads : {2, 4, 8}) {
                double pt = timeIt([&]() { parKMeans(points, initCentroids, k, 100, threads); });
                string label = "Parallel (" + to_string(threads) + " threads)";
                printResult(label, pt);
                cout << "    Speedup: " << fixed << setprecision(2) << seqTime / pt << "x" << endl;
            }
            cout << endl;
        }
    }
    
    // ===== PROBLEM 6: Numerical Integration (Numerical Simulation) =====
    cout << "======================================================================" << endl;
    cout << "  BENCHMARK 6: Numerical Integration — Trapezoidal Rule" << endl;
    cout << "======================================================================\n" << endl;
    
    {
        // Sequential trapezoidal integration
        auto seqIntegrate = [](function<double(double)> f, double a, double b, int n) -> double {
            double h = (b - a) / n;
            double sum = 0.5 * (f(a) + f(b));
            for (int i = 1; i < n; i++)
                sum += f(a + h * i);
            return h * sum;
        };
        
        // Parallel trapezoidal integration (domain decomposition)
        auto parIntegrate = [&seqIntegrate](function<double(double)> f, double a, double b, int n, int numThreads) -> double {
            int subN = n / numThreads;
            double h = (b - a) / numThreads;
            vector<future<double>> futures;
            
            for (int t = 0; t < numThreads; t++) {
                double lo = a + h * t;
                double hi = a + h * (t + 1);
                futures.push_back(async(launch::async, [=, &seqIntegrate]() {
                    return seqIntegrate(f, lo, hi, subN);
                }));
            }
            
            double total = 0;
            for (auto& fut : futures) total += fut.get();
            return total;
        };
        
        struct TestFunc {
            string name;
            function<double(double)> f;
            double a, b, exact;
        };
        
        vector<TestFunc> testFuncs;
        testFuncs.push_back(TestFunc{"sin(x) on [0,pi]", [](double x){ return sin(x); }, 0, M_PI, 2.0});
        testFuncs.push_back(TestFunc{"4/(1+x^2) on [0,1] = pi", [](double x){ return 4.0/(1.0+x*x); }, 0, 1, M_PI});
        testFuncs.push_back(TestFunc{"x^2 on [0,1] = 1/3", [](double x){ return x*x; }, 0, 1, 1.0/3.0});
        
        for (auto& tf : testFuncs) {
            cout << "  === " << tf.name << " ===" << endl;
            
            for (int n : {1000000, 10000000, 50000000}) {
                cout << "  --- N = " << n << " ---" << endl;
                
                double seqResult;
                double seqTime = timeIt([&]() { seqResult = seqIntegrate(tf.f, tf.a, tf.b, n); });
                printResult("Sequential", seqTime);
                cout << "    Result: " << fixed << setprecision(15) << seqResult << endl;
                
                for (int threads : {2, 4, 8}) {
                    double parResult;
                    double pt = timeIt([&]() { parResult = parIntegrate(tf.f, tf.a, tf.b, n, threads); });
                    string label = "Parallel (" + to_string(threads) + " threads)";
                    printResult(label, pt);
                    cout << "    Speedup: " << fixed << setprecision(2) << seqTime / pt << "x" << endl;
                }
                cout << endl;
            }
        }
    }
    
    cout << "\nAll C++ benchmarks complete!" << endl;
    return 0;
}
