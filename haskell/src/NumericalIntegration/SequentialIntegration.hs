{-# LANGUAGE BangPatterns #-}
{-
================================================================================
  NumericalIntegration.SequentialIntegration — Sequential Numerical Integration
================================================================================

PURPOSE:
  Approximate definite integrals using the Trapezoidal Rule. This is a
  fundamental NUMERICAL SIMULATION technique used in physics, engineering,
  and scientific computing — directly relevant to the project requirements.

THE TRAPEZOIDAL RULE:
  To approximate ∫[a,b] f(x) dx:
  1. Divide [a,b] into N equally-spaced sub-intervals of width h = (b-a)/N
  2. For each sub-interval [xᵢ, xᵢ₊₁]:
     Approximate the area as a trapezoid: h × (f(xᵢ) + f(xᵢ₊₁)) / 2
  3. Sum all trapezoid areas

  Mathematically:
    ∫[a,b] f(x) dx ≈ h × [f(a)/2 + f(x₁) + f(x₂) + ... + f(xₙ₋₁) + f(b)/2]

  ACCURACY: O(h²) = O(1/N²) — doubles N → 4× more accurate
  This means we need LOTS of sub-intervals for high accuracy,
  making it a perfect candidate for parallelism.

KEY FP CONCEPTS DEMONSTRATED:

  1. HIGHER-ORDER FUNCTIONS:
     The integrator takes a FUNCTION as an argument:
       integrate :: (Double -> Double) -> Double -> Double -> Int -> Double
     
     This is a textbook example of higher-order programming.
     The same integrator works for ANY function: sin, cos, polynomials, etc.
     
     COMPARISON WITH C++:
       C++: Must use function pointers, std::function, or templates
       Haskell: Functions are first-class values — just pass them as arguments

  2. LIST GENERATION AND REDUCTION:
     We generate the list of x-values using a list comprehension:
       xs = [a + h * fromIntegral i | i <- [0..n]]
     Then reduce (sum) the function evaluations:
       sum $ map f xs
     
     This is a natural MAPREDUCE pattern:
       MAP: evaluate f at each x
       REDUCE: sum the results

  3. STRICT ACCUMULATION:
     We use strict foldl' to accumulate the sum. Without strictness,
     summing millions of terms would build a huge chain of unevaluated
     additions, causing stack overflow.

  4. PURE COMPUTATION:
     The integration is a pure function — no I/O, no state.
     Given the same (f, a, b, n), it always returns the same result.
     This makes it trivially parallelizable.

TEST FUNCTIONS AND KNOWN ANSWERS:
  - ∫[0,π] sin(x) dx = 2.0
  - ∫[0,1] x² dx = 1/3 ≈ 0.33333
  - ∫[0,1] 4/(1+x²) dx = π ≈ 3.14159 (Leibniz formula for π)
  - ∫[0,1] e^(-x²) dx ≈ 0.74682 (Gauss error function related)

================================================================================
-}

module NumericalIntegration.SequentialIntegration
    ( integrate
    , integrateStrict
    , testFunctions
    , TestFunction(..)
    ) where

import Data.List (foldl')

-- ============================================================================
-- Data Types
-- ============================================================================

{-|
  A test function with its integration bounds and known exact answer.
  Used for benchmarking and correctness verification.
-}
data TestFunction = TestFunction
    { funcName    :: String
    , func        :: Double -> Double   -- The function to integrate
    , lowerBound  :: Double             -- Integration lower bound
    , upperBound  :: Double             -- Integration upper bound
    , exactAnswer :: Double             -- Known exact answer (for error checking)
    }

-- ============================================================================
-- Sequential Integration
-- ============================================================================

{-|
  Trapezoidal rule integration (lazy version).

  This version uses a list comprehension to generate all x-values,
  then maps the function over them. Elegant but allocates a list.

  PARAMETERS:
    - f: the function to integrate (HIGHER-ORDER FUNCTION parameter!)
    - a: lower bound
    - b: upper bound
    - n: number of sub-intervals (more = more accurate, more work)

  HIGHER-ORDER FUNCTION DEMO:
    'f' is a function passed as an argument. We can pass:
      integrate sin 0 pi 1000000   -- ∫ sin(x) from 0 to π
      integrate (\x -> x*x) 0 1 1000000  -- ∫ x² from 0 to 1
      integrate (exp . negate . (^2)) 0 1 1000000  -- ∫ e^(-x²)
-}
integrate :: (Double -> Double) -> Double -> Double -> Int -> Double
integrate f a b n =
    let h  = (b - a) / fromIntegral n
        xs = [a + h * fromIntegral i | i <- [0..n]]
        ys = map f xs
        -- First and last terms get weight 1/2 (trapezoidal rule)
        innerSum = sum (tail (init ys))
        endTerms = (head ys + last ys) / 2.0
    in h * (endTerms + innerSum)

{-|
  Trapezoidal rule integration (strict accumulation version).

  This version uses a strict fold instead of building a list.
  Better for large N because it doesn't allocate the intermediate list.

  IMPLEMENTATION:
    We iterate from i=0 to i=n, accumulating the weighted function
    evaluations. The BangPattern !acc ensures strict evaluation,
    preventing thunk accumulation.

  This is the version we'll parallelize, because the loop body
  (function evaluation at each point) is independent and can be
  distributed across cores.
-}
integrateStrict :: (Double -> Double) -> Double -> Double -> Int -> Double
integrateStrict f a b n =
    let -- Strict fold over the sub-interval indices
        !total = foldl' accumulate 0.0 [0..n]
    in h * total
  where
    h = (b - a) / fromIntegral n
    accumulate :: Double -> Int -> Double
    accumulate !acc i =
        let x = a + h * fromIntegral i
            weight = if i == 0 || i == n then 0.5 else 1.0
        in acc + weight * f x

-- ============================================================================
-- Test Functions (with known exact answers)
-- ============================================================================

{-|
  A collection of test functions with known integrals.
  Used for both benchmarking and verifying correctness.

  FP CONCEPT: Higher-order data — we store FUNCTIONS in a data structure!
  This would be awkward in C (function pointers) but natural in Haskell.
-}
testFunctions :: [TestFunction]
testFunctions =
    [ TestFunction
        { funcName    = "sin(x) on [0,π]"
        , func        = sin
        , lowerBound  = 0
        , upperBound  = pi
        , exactAnswer = 2.0
        }
    , TestFunction
        { funcName    = "4/(1+x²) on [0,1] = π"
        , func        = \x -> 4.0 / (1.0 + x * x)
        , lowerBound  = 0
        , upperBound  = 1
        , exactAnswer = pi
        }
    , TestFunction
        { funcName    = "x² on [0,1] = 1/3"
        , func        = \x -> x * x
        , lowerBound  = 0
        , upperBound  = 1
        , exactAnswer = 1.0 / 3.0
        }
    , TestFunction
        { funcName    = "e^(-x²) on [0,1]"
        , func        = \x -> exp (-(x * x))
        , lowerBound  = 0
        , upperBound  = 1
        , exactAnswer = 0.746824132812427  -- erf(1) * sqrt(pi)/2 
        }
    ]
