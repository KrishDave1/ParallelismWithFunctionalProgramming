{-
================================================================================
  Sorting.ParallelMergeSort — Parallel Merge Sort using Eval Monad & Strategies
================================================================================

PURPOSE:
  This module implements a PARALLEL merge sort using Haskell's Eval monad
  and evaluation strategies. It demonstrates how purity enables safe
  parallelism with minimal code changes.

KEY FP CONCEPTS DEMONSTRATED:

  1. EVAL MONAD & SPARKS (rpar / rseq):
     Haskell's primary mechanism for SEMI-EXPLICIT parallelism.
     
     - 'rpar x' creates a "spark" — a lightweight hint to the runtime
       that 'x' can be evaluated in parallel. The RTS decides whether
       to actually run it on a separate core.
     - 'rseq x' forces sequential evaluation (wait for result).
     - The Eval monad sequences these operations.
     
     ANALOGY: Think of sparks as "todo items" placed on a work-stealing
     queue. Worker threads on other cores can pick them up.

  2. STRATEGIES (rdeepseq, parList, etc.):
     Higher-level abstractions built on top of rpar/rseq.
     A Strategy is a function that specifies HOW to evaluate a data structure.
     
     @using :: a -> Strategy a -> a@
     
     Separates the ALGORITHM from the PARALLELISM STRATEGY.
     You can change parallelism behavior WITHOUT changing the algorithm.

  3. DEPTH THRESHOLD:
     Creating a spark for every recursive call is wasteful — the overhead
     of spark creation exceeds the benefit for small sublists. We use a
     depth threshold: parallelize only the top 'k' levels of recursion.
     
     With depth threshold = log₂(numCores), we create roughly as many
     parallel tasks as cores — optimal for work distribution.

  4. PURITY → SAFE PARALLELISM:
     Because 'mergeSort' is a PURE function (no side effects, no mutation),
     it is REFERENTIALLY TRANSPARENT. This means:
     - Evaluating the left half CANNOT affect the right half
     - No data races, no locks, no synchronization needed
     - The parallel version produces the EXACT same result as sequential
     
     THIS IS THE CENTRAL THESIS: Purity makes parallelism safe and easy.

  5. LAZY EVALUATION INTERACTION:
     Haskell is lazy by default — expressions are only evaluated when needed.
     For parallelism, this is both a blessing and a curse:
     - Blessing: Sparks are cheap because they're just thunks (deferred computations)
     - Curse: Without 'force'/'rdeepseq', a spark might only evaluate to WHNF
       (just the first constructor), not the whole list
     We use 'rdeepseq' to ensure full evaluation of each sorted sublist.

IMPLEMENTATION:
  The parallel merge sort is identical to sequential EXCEPT that the two
  recursive calls are sparked for parallel evaluation:

  SEQUENTIAL:                          PARALLEL:
    let left  = mergeSort xs           runEval $ do
        right = mergeSort ys               left  <- rpar (force $ mergeSort xs)
    in merge left right                    right <- rseq (force $ mergeSort ys)
                                           return (merge left right)

  That's it. Four extra lines to go from sequential to parallel.
  In C++, the equivalent change requires threads, futures, or OpenMP pragmas,
  plus careful handling of shared memory.

SPARK STATISTICS (from +RTS -s):
  When you run with: +RTS -N4 -s
  You'll see output like:
    SPARKS: 64 (40 converted, 0 overflowed, 0 dud, 4 GC'd, 20 fizzled)
  
  - converted: Actually ran in parallel on another core ✓
  - fizzled: The main thread evaluated it before a worker got to it
  - GC'd: Spark was garbage collected before evaluation
  - dud: Spark was already evaluated (redundant)
  
  GOAL: Maximize "converted" sparks. If most sparks fizzle, the work
  granularity is too fine (lower the depth threshold).

================================================================================
-}

module Sorting.ParallelMergeSort
    ( parallelMergeSort
    , parallelMergeSortWithDepth
    ) where

import Sorting.SequentialMergeSort (merge, mergeSort)
import Control.Parallel.Strategies (rpar, rseq, rdeepseq, using, Strategy, runEval)
import Control.DeepSeq (NFData, force)

{-|
  Parallel merge sort using Eval monad with a default depth threshold.

  DEFAULT DEPTH = 4, which creates up to 2^4 = 16 parallel tasks.
  This works well for 4-8 core machines.

  USAGE:
    let sorted = parallelMergeSort [5, 3, 1, 4, 2]
    -- sorted == [1, 2, 3, 4, 5]

  TO BENCHMARK:
    Run with: +RTS -N4 -s    (4 cores, show statistics)
-}
parallelMergeSort :: (Ord a, NFData a) => [a] -> [a]
parallelMergeSort = parallelMergeSortWithDepth 4

{-|
  Parallel merge sort with configurable depth threshold.

  PARAMETERS:
    - maxDepth: how many levels of recursion to parallelize
      - Each level doubles the number of parallel tasks: 2^maxDepth total
      - maxDepth = 0 → fully sequential (no parallelism)
      - maxDepth = 3 → 8 parallel tasks  (good for 4 cores)
      - maxDepth = 4 → 16 parallel tasks (good for 8 cores)
      - maxDepth = 10 → 1024 tasks (too many! overhead dominates)

  IMPLEMENTATION WALKTHROUGH:
    1. Base cases: lists of length 0 or 1 are already sorted
    2. Split the list in half
    3. IF depth > 0 (should parallelize):
       a. Create a spark for the left half:  rpar (force $ recurse left)
       b. Sequentially sort the right half:  rseq (force $ recurse right)
       c. Merge the results
    4. IF depth == 0 (switch to sequential):
       Fall back to sequential mergeSort to avoid spark overhead

  WHY 'force'?
    'rpar' by default evaluates to Weak Head Normal Form (WHNF).
    For a list, WHNF means just the first (:) constructor.
    We need the ENTIRE sorted list evaluated, so we use 'force'
    to evaluate to Normal Form (NF) before the spark is "done".
    
    Without 'force': spark finishes after evaluating "1 : <thunk>"
    With 'force':    spark finishes after evaluating "[1,2,3,4,5]"
-}
parallelMergeSortWithDepth :: (Ord a, NFData a) => Int -> [a] -> [a]
parallelMergeSortWithDepth _ []  = []
parallelMergeSortWithDepth _ [x] = [x]
parallelMergeSortWithDepth depth xs
    -- PARALLEL PATH: we still have depth budget, so evaluate halves in parallel
    | depth > 0 =
        let mid            = length xs `div` 2
            (left, right)  = splitAt mid xs
        in runEval $ do
            -- Spark the left half for parallel evaluation
            -- 'rpar' places this on the spark pool for a worker thread to pick up
            sortedLeft  <- rpar (force $ parallelMergeSortWithDepth (depth - 1) left)
            -- Evaluate the right half on the current thread (sequential)
            -- 'rseq' means "evaluate this here and now"
            sortedRight <- rseq (force $ parallelMergeSortWithDepth (depth - 1) right)
            -- Merge the results — both halves are now fully evaluated
            return (merge sortedLeft sortedRight)

    -- SEQUENTIAL PATH: depth exhausted, no more parallelism overhead
    -- Falls back to the simple sequential merge sort
    | otherwise = mergeSort xs
