{-# LANGUAGE BangPatterns #-}
{-
================================================================================
  MonteCarlo.SequentialPi — Sequential Monte Carlo Pi Estimation (Baseline)
================================================================================

PURPOSE:
  Estimate the value of π using the Monte Carlo method.
  This is a classic example of an "embarrassingly parallel" problem —
  each random sample is completely independent of every other sample.

THE ALGORITHM:
  1. Imagine a unit square [0,1] × [0,1] with a quarter circle of radius 1
     inscribed in it (centered at origin).
  
  2. Generate N random (x,y) points uniformly in the square.
  
  3. Count how many fall inside the quarter circle: x² + y² ≤ 1
  
  4. The ratio (points inside / total points) ≈ π/4
     Therefore: π ≈ 4 × (points inside / total points)
  
  WHY?
    Area of quarter circle = π × r² / 4 = π/4 (for r=1)
    Area of unit square = 1
    Ratio = π/4

KEY FP CONCEPTS DEMONSTRATED:

  1. PURE RANDOM NUMBER GENERATION:
     In imperative languages, random number generation uses hidden mutable state
     (the PRNG seed is a global variable that gets mutated on each call).
     
     In Haskell, we use SPLITTABLE RNGs (SplitMix):
     - 'split gen' returns TWO independent generators from one
     - No mutable state, no global variables
     - Each generator is a pure value that can be passed to threads
     - CRITICAL for parallelism: each thread gets its own generator,
       no synchronization needed for random number generation
     
     COMPARISON WITH C++:
       C++: std::mt19937 gen(seed); // mutable, shared, needs locks for threads
       Haskell: let (gen1, gen2) = split gen  // pure, independent, no locks

  2. HIGHER-ORDER FUNCTIONS FOR ACCUMULATION:
     We use 'foldl'' to count hits in a single pass:
       foldl' (\count (x,y) -> if x*x + y*y <= 1 then count+1 else count) 0 samples
     
     No loop variables, no indices, no off-by-one errors.

  3. REFERENTIAL TRANSPARENCY:
     Given the same seed, estimatePi ALWAYS returns the same result.
     This makes debugging and testing easy — crucial for a benchmark.

ACCURACY:
  The Monte Carlo method converges as O(1/√n):
    n = 1,000     → ~2 decimal places of π
    n = 100,000   → ~3 decimal places
    n = 10,000,000 → ~4 decimal places
    n = 1,000,000,000 → ~5 decimal places

================================================================================
-}

module MonteCarlo.SequentialPi
    ( estimatePi
    , estimatePiVerbose
    , isInsideCircle
    , countHits
    ) where

import System.Random.SplitMix (SMGen, mkSMGen, nextDouble)

import Data.Word (Word64)

{-|
  Estimate π using the sequential Monte Carlo method.

  PARAMETERS:
    - seed: PRNG seed for reproducibility
    - n:    number of random samples

  RETURNS:
    Estimated value of π

  IMPLEMENTATION:
    We generate pairs of (x, y) coordinates and count how many
    fall inside the unit quarter circle. We use SplitMix as our
    PRNG because it's fast and has good statistical properties.

  NOTE ON STRICT ACCUMULATION:
    We use a strict accumulator to avoid building up thunks.
    Without strictness, counting 100 million samples would build
    a chain of 100 million unevaluated additions → stack overflow.
-}
estimatePi :: Word64 -> Int -> Double
estimatePi seed n =
    let gen = mkSMGen seed
        hits = countHits n gen
    in 4.0 * fromIntegral hits / fromIntegral n

{-|
  Same as estimatePi but also returns the hit count for debugging.
-}
estimatePiVerbose :: Word64 -> Int -> (Double, Int, Int)
estimatePiVerbose seed n =
    let gen  = mkSMGen seed
        hits = countHits n gen
        piEst = 4.0 * fromIntegral hits / fromIntegral n
    in (piEst, hits, n)

{-|
  Check if a point (x,y) falls inside the unit quarter circle.
  This is a pure function — no side effects, no state.
-}
isInsideCircle :: Double -> Double -> Bool
isInsideCircle x y = x * x + y * y <= 1.0

{-|
  Count the number of random points that fall inside the quarter circle.

  IMPLEMENTATION:
    We manually thread the PRNG state through each iteration.
    Each call to 'nextDouble' returns (value, newGenerator).
    
    This is a common FP pattern: instead of mutating a global PRNG,
    we explicitly pass the state through the computation.
    
    An alternative would be to use the State monad to hide this
    threading, but explicit passing is clearer for educational purposes.
-}
countHits :: Int -> SMGen -> Int
countHits n gen0 = go n gen0 0
  where
    go :: Int -> SMGen -> Int -> Int
    go 0 _   !acc = acc  -- !acc forces strict evaluation (BangPattern)
    go k gen !acc =
        let (x, gen')  = nextDouble gen      -- Generate x coordinate
            (y, gen'') = nextDouble gen'     -- Generate y coordinate
            acc'       = if isInsideCircle x y
                         then acc + 1        -- Inside circle: increment
                         else acc            -- Outside: keep count
        in go (k - 1) gen'' acc'
