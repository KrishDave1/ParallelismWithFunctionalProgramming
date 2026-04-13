{-
================================================================================
  Matrix.SequentialMatMul — Sequential Matrix Multiplication (Baseline)
================================================================================

PURPOSE:
  Standard O(n³) matrix multiplication implemented using pure Haskell lists.
  This serves as our sequential baseline for the matrix multiplication benchmark.

KEY FP CONCEPTS DEMONSTRATED:

  1. LIST COMPREHENSIONS:
     Haskell's list comprehensions provide a declarative way to express
     matrix operations that reads almost like mathematical notation:
       C[i][j] = Σ A[i][k] * B[k][j]
     becomes:
       [[ sum $ zipWith (*) row col | col <- transpose b ] | row <- a ]
     
     Compare with C++: three nested for-loops with index manipulation.

  2. HIGHER-ORDER FUNCTIONS (zipWith, map, transpose):
     - 'zipWith (*)' pairs up corresponding elements and multiplies them
     - 'map' applies a function to every row/column
     - 'transpose' flips rows and columns
     These compose naturally: no temporary variables, no index bookkeeping.

  3. IMMUTABLE DATA:
     The input matrices are never modified. Each operation creates new lists.
     This is "wasteful" compared to C++ in-place multiplication, but:
     - Correctness is trivial to verify
     - No aliasing bugs (where output matrix = input matrix)
     - Safe to share matrices across threads without copying

REPRESENTATION:
  We represent matrices as [[Double]] — a list of rows, where each row is
  a list of Doubles. This is simple but not cache-friendly.
  The parallel version will use Vector for better performance.

PERFORMANCE NOTE:
  List-based matrix multiplication is SLOW compared to C++/NumPy because:
  1. Lists are linked lists → poor cache locality
  2. Each element is boxed (wrapped in a heap object)
  3. No SIMD vectorization
  However, this makes the speedup from parallelism even more dramatic,
  and the comparison with C++ more interesting for analysis.

================================================================================
-}

module Matrix.SequentialMatMul
    ( matMul
    , transpose'
    , matSize
    , matEqual
    ) where

import Control.DeepSeq (NFData(..))

{-|
  Sequential matrix multiplication: C = A × B

  MATHEMATICAL DEFINITION:
    C[i][j] = Σₖ A[i][k] × B[k][j]

  HASKELL IMPLEMENTATION:
    For each row of A, compute its dot product with each column of B.
    We get columns of B by transposing B first.

  STEP BY STEP:
    1. Transpose B so we can access columns as rows: B^T
    2. For each row in A:
       3. For each row in B^T (= column of B):
          4. Compute dot product: sum of element-wise multiplication
    
  This reads almost like the math formula — that's the power of FP!
-}
matMul :: [[Double]] -> [[Double]] -> [[Double]]
matMul a b =
    let bt = transpose' b  -- Transpose B for column access
    in [[ dotProduct row col | col <- bt ] | row <- a]
  where
    -- Dot product of two vectors using higher-order functions:
    -- zipWith (*) pairs elements and multiplies them
    -- sum adds up all the products
    dotProduct :: [Double] -> [Double] -> Double
    dotProduct xs ys = sum $ zipWith (*) xs ys

{-|
  Transpose a matrix: swap rows and columns.

  EXAMPLE:
    transpose' [[1,2,3], [4,5,6]] = [[1,4], [2,5], [3,6]]

  IMPLEMENTATION:
    - Base case: if any row is empty, we're done
    - Otherwise: take the first element of each row (= first column)
      and recurse on the remaining elements

  This is a standard FP pattern for working with nested lists.
-}
transpose' :: [[a]] -> [[a]]
transpose' [] = []
transpose' ([] : _) = []
transpose' xss = map head xss : transpose' (map tail xss)

{-|
  Get the dimensions of a matrix: (rows, cols)
-}
matSize :: [[a]] -> (Int, Int)
matSize [] = (0, 0)
matSize m  = (length m, length (head m))

{-|
  Check if two matrices are approximately equal (within epsilon).
  Useful for verifying that parallel results match sequential results.
-}
matEqual :: Double -> [[Double]] -> [[Double]] -> Bool
matEqual eps a b =
    length a == length b &&
    all (\(row1, row2) ->
        length row1 == length row2 &&
        all (\(x, y) -> abs (x - y) < eps) (zip row1 row2)
    ) (zip a b)
