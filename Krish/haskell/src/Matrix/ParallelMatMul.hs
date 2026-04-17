{-
================================================================================
  Matrix.ParallelMatMul — Parallel Matrix Multiplication
================================================================================

PURPOSE:
  Parallel matrix multiplication using two approaches:
    1. Strategy-based parallelism (parMap over rows)
    2. Manual chunking with Eval monad

  This demonstrates DATA PARALLELISM — the same operation (compute one row
  of the result) applied independently to different data (different rows).

KEY FP CONCEPTS DEMONSTRATED:

  1. DATA PARALLELISM via Strategies:
     'parMap rdeepseq f xs' applies 'f' to each element of 'xs' in parallel.
     This is the simplest form of parallelism in Haskell:
     - Replace 'map f xs' with 'parMap rdeepseq f xs'
     - Done! The runtime handles work distribution.
     
     Compare with C++ OpenMP:
       #pragma omp parallel for
       for (int i = 0; i < n; i++) { ... }
     
     The Haskell version is MORE composable: parMap is just a function,
     not a compiler directive. You can pass it around, abstract over it,
     compose it with other parallel strategies.

  2. parList STRATEGY:
     Instead of modifying the algorithm, we apply a STRATEGY to the result:
       result `using` parList rdeepseq
     
     This separates the WHAT (matrix multiplication) from the HOW (parallel
     evaluation). We can switch parallelism strategies without changing the
     algorithm — a key software engineering benefit.
     
     Available strategies include:
     - rseq:      evaluate sequentially
     - rpar:      spark for parallel evaluation  
     - rdeepseq:  fully evaluate in parallel
     - parList:   evaluate all list elements in parallel
     - parListChunk: evaluate in chunks (reduces spark overhead)

  3. CHUNKED PARALLELISM:
     Instead of sparking one task per row, we group rows into chunks.
     This reduces spark overhead while maintaining good load balance.
     
     chunk size = numRows / numCores
     
     Each chunk creates one spark, matching the number of available cores.

COMPARISON WITH IMPERATIVE PARALLEL MATRIX MULTIPLY:
  
  C++ with OpenMP:
    - Add '#pragma omp parallel for' before the outer loop
    - Threads share the output matrix (write to different rows → safe)
    - Must be careful about false sharing if rows are small
    - Compiler-specific pragma, not portable to other parallelism models
  
  Haskell with Strategies:
    - Replace 'map' with 'parMap rdeepseq'
    - OR: add 'using parList rdeepseq' to the result
    - No shared mutable state — each row is independently computed
    - Strategy is a first-class value, composable and portable

================================================================================
-}

module Matrix.ParallelMatMul
    ( parallelMatMul
    , parallelMatMulChunked
    ) where

import Matrix.SequentialMatMul (transpose')
import Control.Parallel.Strategies (rdeepseq, parMap, parList, using)

{-|
  Parallel matrix multiplication using parMap.

  HOW IT WORKS:
    1. Transpose B (same as sequential)
    2. For each row of A, compute its dot product with every column of B
       BUT: use 'parMap' instead of 'map' so rows are computed in parallel
    
  CHANGE FROM SEQUENTIAL (just ONE word changed!):
    Sequential: map    (\row -> [dotProduct row col | col <- bt]) a
    Parallel:   parMap rdeepseq (\row -> [dotProduct row col | col <- bt]) a
    
  'rdeepseq' ensures each row is FULLY evaluated (all elements computed)
  before the spark is considered complete.

  This is DATA PARALLELISM: same operation, different data, no dependencies.
-}
parallelMatMul :: [[Double]] -> [[Double]] -> [[Double]]
parallelMatMul a b =
    let bt = transpose' b
        -- Compute each row of the result in parallel
        -- parMap rdeepseq creates one spark per row
    in parMap rdeepseq (computeRow bt) a
  where
    computeRow :: [[Double]] -> [Double] -> [Double]
    computeRow bt' row = [dotProduct row col | col <- bt']
    
    dotProduct :: [Double] -> [Double] -> Double
    dotProduct xs ys = sum $ zipWith (*) xs ys

{-|
  Parallel matrix multiplication with explicit chunking.

  WHY CHUNKING?
    With a 1024×1024 matrix, parMap creates 1024 sparks.
    Each spark has overhead (allocation, scheduling). For small rows,
    this overhead can dominate.
    
    CHUNKING groups rows together:
    - 1024 rows ÷ 8 cores = 128 rows per chunk
    - Only 8 sparks created (one per core) → minimal overhead
    - Each chunk does 128 rows sequentially → good cache behavior

  IMPLEMENTATION:
    1. Split rows into chunks of size (nRows / numChunks)
    2. Use 'parList rdeepseq' to evaluate all chunks in parallel
    3. Each chunk internally uses 'map' (sequential within chunk)
    4. Concatenate results

  PARAMETER:
    - numChunks: number of chunks (ideally = number of cores)
-}
parallelMatMulChunked :: Int -> [[Double]] -> [[Double]] -> [[Double]]
parallelMatMulChunked numChunks a b =
    let bt          = transpose' b
        chunkSize   = max 1 (length a `div` numChunks)
        rowChunks   = chunksOf chunkSize a
        -- Process each chunk in parallel, sequential within each chunk
        resultChunks = map (map (computeRow bt)) rowChunks
                       `using` parList rdeepseq
    in concat resultChunks
  where
    computeRow :: [[Double]] -> [Double] -> [Double]
    computeRow bt' row = [dotProduct row col | col <- bt']
    
    dotProduct :: [Double] -> [Double] -> Double
    dotProduct xs ys = sum $ zipWith (*) xs ys
    
    -- Split a list into chunks of size n
    chunksOf :: Int -> [e] -> [[e]]
    chunksOf _ [] = []
    chunksOf n xs = let (chunk, rest) = splitAt n xs in chunk : chunksOf n rest
