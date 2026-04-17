{-# LANGUAGE BangPatterns #-}
{-
================================================================================
  NumericalIntegration.ParallelIntegration — Parallel Numerical Integration
================================================================================

PURPOSE:
  Parallel numerical integration using two approaches:
    1. Eval monad with rpar/rseq (divide domain into sub-ranges)
    2. Strategies with parListChunk (parallel map over evaluation points)

  This demonstrates DOMAIN DECOMPOSITION parallelism — splitting the
  integration domain [a,b] into sub-ranges and computing each sub-integral
  independently on different cores.

KEY FP CONCEPTS DEMONSTRATED:

  1. DOMAIN DECOMPOSITION (Divide the Problem, Not the Data):
     Unlike data parallelism (where we split DATA), here we split the DOMAIN.
     
     Sequential: compute ∫[a,b] f(x) dx as one big sum
     Parallel:   split [a,b] into [a,m] and [m,b]
                 compute ∫[a,m] f(x) dx on core 1
                 compute ∫[m,b] f(x) dx on core 2
                 add results
     
     This works because integration is ADDITIVE:
       ∫[a,b] f(x) dx = ∫[a,m] f(x) dx + ∫[m,b] f(x) dx
     
     This algebraic property (linearity of integration) is what makes
     the parallelism mathematically correct.

  2. HIGHER-ORDER FUNCTION + PARALLELISM:
     We pass the function 'f' to the integrator, and the integrator
     parallelizes the evaluation of 'f'. The function itself has no
     knowledge of parallelism — it's just a pure (Double -> Double).
     
     This is SEPARATION OF CONCERNS:
     - The FUNCTION author doesn't need to think about parallelism
     - The INTEGRATOR handles parallelism
     - They compose cleanly because 'f' is pure

  3. COMPARING TWO PARALLELISM APPROACHES:
     
     APPROACH 1: Eval monad (fine-grained control)
       - Explicitly spark sub-range computations
       - Depth-based recursion (like parallel merge sort)
       - Good for divide-and-conquer problems
     
     APPROACH 2: parListChunk (coarse-grained, simpler)
       - Generate all evaluation points
       - Evaluate function at each point in parallel chunks
       - Simpler code, less control over granularity
     
     We implement both to show the spectrum of Haskell's parallelism tools.

  4. NUMERICAL SIMULATION APPLICATION:
     This directly addresses the project requirement for
     "numerical simulations suitable for parallel processing."
     Integration is foundational to:
     - Physics simulations (computing forces, energies)
     - Financial modeling (option pricing)
     - Signal processing (Fourier transforms)
     - Machine learning (computing expectations)

COMPARISON WITH IMPERATIVE APPROACHES:
  
  C++ with OpenMP:
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        double x = a + h * i;
        sum += f(x) * weight;
    }
    // Simple but OpenMP-specific, not composable
  
  Haskell with Strategies:
    sums = map (\(lo,hi) -> integrateStrict f lo hi subN) ranges
           `using` parList rdeepseq
    total = sum sums
    // Strategy is a first-class value, composable, portable

================================================================================
-}

module NumericalIntegration.ParallelIntegration
    ( parallelIntegrate
    , parallelIntegrateChunked
    ) where

import NumericalIntegration.SequentialIntegration (integrateStrict)
import Control.Parallel.Strategies
    ( rdeepseq, parList, using )

-- ============================================================================
-- Approach 1: Domain Decomposition with Eval Monad
-- ============================================================================

{-|
  Parallel integration via domain decomposition.

  STRATEGY:
    1. Split [a,b] into 'numChunks' equal sub-ranges
    2. Compute each sub-integral in parallel using rpar
    3. Sum the sub-integrals

  HOW IT WORKS (example with 4 chunks on [0, 1]):
    Chunk 1: ∫[0.00, 0.25] f(x) dx → sparked
    Chunk 2: ∫[0.25, 0.50] f(x) dx → sparked
    Chunk 3: ∫[0.50, 0.75] f(x) dx → sparked
    Chunk 4: ∫[0.75, 1.00] f(x) dx → evaluated locally

  PARAMETERS:
    - f:         the function to integrate
    - a, b:      integration bounds
    - n:         total number of sub-intervals across entire domain
    - numChunks: number of parallel tasks (≈ numCores)

  MATHEMATICAL CORRECTNESS:
    ∫[a,b] f(x) dx = Σᵢ ∫[aᵢ, aᵢ₊₁] f(x) dx
    Each sub-integral uses n/numChunks sub-intervals of the trapezoidal rule.
    The sum of sub-integrals equals the full integral (up to floating point).
-}
parallelIntegrate :: (Double -> Double) -> Double -> Double -> Int -> Int -> Double
parallelIntegrate f a b n numChunks =
    let subN     = max 1 (n `div` numChunks)  -- sub-intervals per chunk
        h        = (b - a) / fromIntegral numChunks
        -- Generate (lo, hi) ranges for each chunk
        ranges   = [(a + h * fromIntegral i, a + h * fromIntegral (i + 1))
                    | i <- [0 .. numChunks - 1]]
        -- Compute each sub-integral
        subResults = map (\(lo, hi) -> integrateStrict f lo hi subN) ranges
        -- Evaluate all sub-integrals in parallel
        parResults = subResults `using` parList rdeepseq
    in sum parResults

-- ============================================================================
-- Approach 2: Parallel Evaluation with Chunking
-- ============================================================================

{-|
  Alternative parallel integration using parList with explicit chunking.

  This approach generates all evaluation points first, then evaluates
  the function at each point in parallel chunks.

  DIFFERENCE FROM APPROACH 1:
    Approach 1: Domain decomposition — each chunk integrates a sub-range
    Approach 2: Point evaluation — each chunk evaluates f at some points

  Approach 1 is generally better because:
  1. Less overhead (fewer function calls to parList)
  2. Better cache locality (each chunk works on contiguous x-values)
  3. More natural mapping to cores

  But Approach 2 shows a different way to think about the parallelism.
-}
parallelIntegrateChunked :: (Double -> Double) -> Double -> Double -> Int -> Int -> Double
parallelIntegrateChunked f a b n numChunks =
    let h         = (b - a) / fromIntegral n
        -- Generate weighted function evaluations for each point
        -- endpoints get weight 0.5 (trapezoidal rule)
        evals     = [ let x      = a + h * fromIntegral i
                          weight = if i == 0 || i == n then 0.5 else 1.0
                      in weight * f x
                    | i <- [0..n]
                    ]
        -- Evaluate in parallel chunks
        chunkSize = max 1 ((n + 1) `div` numChunks)
        chunked   = chunksOf chunkSize evals
        parSums   = map sum chunked `using` parList rdeepseq
    in h * sum parSums
  where
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = let (chunk, rest) = splitAt k xs in chunk : chunksOf k rest
