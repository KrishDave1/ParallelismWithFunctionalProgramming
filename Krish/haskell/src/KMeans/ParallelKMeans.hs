{-# LANGUAGE BangPatterns #-}
{-
================================================================================
  KMeans.ParallelKMeans — Parallel K-Means Clustering
================================================================================

PURPOSE:
  Parallel K-Means clustering using Haskell's Strategies.
  The ASSIGN step — where each point finds its nearest centroid — is
  embarrassingly parallel. We parallelize it using 'parListChunk'.

KEY FP CONCEPTS DEMONSTRATED:

  1. STRATEGY-BASED PARALLELISM (Separating Algorithm from Parallelism):
     The sequential K-Means code:
       assignments = map (assignPoint centroids) points
     
     The parallel K-Means code:
       assignments = map (assignPoint centroids) points `using` parListChunk chunkSize rdeepseq
     
     The ALGORITHM is UNCHANGED. We only add a strategy that tells the
     runtime HOW to evaluate the list. This separation is a key advantage
     of Haskell's approach:
     - You can swap strategies without changing the algorithm
     - You can test with sequential evaluation, then add parallelism
     - Strategies are composable and reusable

  2. CHUNKED EVALUATION (parListChunk):
     Instead of sparking one task per point (too much overhead for millions
     of points), we group points into chunks. Each chunk is evaluated as
     one parallel task.
     
     chunkSize = totalPoints / numWorkers
     
     This reduces spark overhead while maintaining good load balance.
     The optimal chunk size ≈ n / (4 × numCores) to allow work stealing.

  3. WHY K-MEANS IS PERFECT FOR FP PARALLELISM:
     a. ASSIGN step: map over independent points → parMap / parListChunk
     b. UPDATE step: fold over assignments → reduction (could parallelize)
     c. No shared mutable state: centroids are recomputed, not mutated
     d. Each iteration is pure: old centroids → new centroids
     
     CONTRAST WITH IMPERATIVE PARALLEL K-MEANS:
       C++/Python: Must use locks/atomics to update shared centroid accumulators
       Haskell:    Each iteration produces a NEW list of centroids (no locks)

  4. CONVERGENCE IS AUTOMATICALLY DETERMINISTIC:
     Because our parallel K-Means uses the same pure functions as sequential,
     it converges to the EXACT SAME result for the same initial centroids.
     No non-determinism from thread scheduling, lock ordering, or race conditions.

ARCHITECTURE:

    ┌─────────────────────────────────────────────────────────┐
    │  Iteration i                                            │
    │                                                         │
    │  centroids_i                                            │
    │       │                                                 │
    │       ▼                                                 │
    │  ┌─────────────────────────────────────┐                │
    │  │ ASSIGN (parallel)                    │                │
    │  │ points split into chunks             │                │
    │  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │                │
    │  │ │Chunk1│ │Chunk2│ │Chunk3│ │ChunkN│ │  parListChunk  │
    │  │ │assign│ │assign│ │assign│ │assign│ │                │
    │  │ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ │                │
    │  └────┼────────┼────────┼────────┼─────┘                │
    │       │        │        │        │                       │
    │       ▼        ▼        ▼        ▼                       │
    │  ┌─────────────────────────────────────┐                │
    │  │ UPDATE (sequential reduction)        │                │
    │  │ Fold assignments → new centroids     │                │
    │  └───────────────┬─────────────────────┘                │
    │                  │                                       │
    │                  ▼                                       │
    │           centroids_{i+1}                                │
    │                  │                                       │
    │           converged? ──no──→ iterate again                │
    │                  │yes                                     │
    │                  ▼                                       │
    │            RESULT                                        │
    └─────────────────────────────────────────────────────────┘

================================================================================
-}

module KMeans.ParallelKMeans
    ( parallelKMeans
    ) where

import KMeans.SequentialKMeans
    ( Point, Centroid, assignPoint, updateCentroids, hasConverged )
import Control.Parallel.Strategies
    ( rdeepseq, parListChunk, using )


{-|
  Parallel K-Means clustering.

  PARAMETERS:
    - chunkSize:     how many points per parallel task
    - maxIter:       maximum iterations
    - epsilon:       convergence threshold
    - initCentroids: initial centroid positions
    - points:        data points to cluster

  RETURNS:
    (final centroids, number of iterations used)

  THE KEY CHANGE FROM SEQUENTIAL:
    Sequential: assignments = map (assignPoint centroids) points
    Parallel:   assignments = map (assignPoint centroids) points
                              `using` parListChunk chunkSize rdeepseq
    
    That's it. ONE LINE added. The algorithm is unchanged.
    
  CHUNK SIZE GUIDANCE:
    - Too small (1): One spark per point → overhead dominates
    - Too large (n): No parallelism → sequential
    - Sweet spot: n / (4 * numCores) — allows work stealing
    - For 100,000 points on 4 cores: chunkSize ≈ 6250
-}
parallelKMeans :: Int       -- ^ chunk size for parallel evaluation
               -> Int       -- ^ max iterations
               -> Double    -- ^ convergence epsilon
               -> [Centroid] -- ^ initial centroids
               -> [Point]   -- ^ data points
               -> ([Centroid], Int) -- ^ (final centroids, iterations used)
parallelKMeans chunkSize maxIter epsilon initCentroids points =
    go maxIter initCentroids 0
  where
    go :: Int -> [Centroid] -> Int -> ([Centroid], Int)
    go 0 centroids iters = (centroids, iters)
    go n centroids !iters =
        let -- PARALLEL ASSIGN: evaluate point assignments in chunks
            -- 'using parListChunk chunkSize rdeepseq' tells the runtime:
            --   "Split this list into chunks of 'chunkSize' elements,
            --    spark each chunk for parallel evaluation,
            --    fully evaluate each element (rdeepseq) within the chunk."
            assignments = map (assignPoint centroids) points
                          `using` parListChunk chunkSize rdeepseq
            
            -- UPDATE: recompute centroids (sequential — fast reduction)
            newCentroids = updateCentroids (length centroids) assignments
            
        in if hasConverged epsilon centroids newCentroids
           then (newCentroids, iters + 1)
           else go (n - 1) newCentroids (iters + 1)
