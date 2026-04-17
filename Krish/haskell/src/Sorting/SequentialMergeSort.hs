{-
================================================================================
  Sorting.SequentialMergeSort — Sequential Merge Sort (Baseline)
================================================================================

PURPOSE:
  This module implements a standard sequential merge sort in Haskell.
  It serves as our BASELINE for measuring parallel speedup.

  Speedup = Time(Sequential) / Time(Parallel)

  A good parallel implementation should achieve speedup > 1, approaching
  the number of cores used (ideal linear speedup).

KEY FP CONCEPTS DEMONSTRATED:
  1. RECURSION: Merge sort is naturally recursive — divide, sort halves, merge.
     In FP, recursion replaces imperative loops.
  
  2. PATTERN MATCHING: We use pattern matching to handle base cases elegantly
     (empty list, single element) rather than if-else chains.
  
  3. IMMUTABLE DATA: Each step creates NEW lists rather than mutating in place.
     This means:
       - No index errors or buffer overflows
       - Easy to reason about correctness
       - BUT: more memory allocation than in-place imperative sort
  
  4. HIGHER-ORDER POTENTIAL: We could easily make this generic over any Ord type
     by using type class constraints. The same code sorts Ints, Strings, etc.

ALGORITHM:
  1. If list has 0 or 1 elements → already sorted (base case)
  2. Split list into two halves
  3. Recursively sort each half
  4. Merge the two sorted halves

TIME COMPLEXITY:  O(n log n) — same as imperative merge sort
SPACE COMPLEXITY: O(n) — Haskell lists are linked lists, so merge creates new nodes

================================================================================
-}

module Sorting.SequentialMergeSort
    ( mergeSort
    , merge
    ) where

import Control.DeepSeq (NFData(..))

{-|
  Sequential merge sort on lists.

  IMPLEMENTATION NOTES:
  - We use 'splitAt' to split the list, which traverses half the list (O(n/2))
  - This is a stable sort: equal elements maintain their relative order
  - Pattern matching makes the base cases clear and exhaustive

  COMPARISON WITH IMPERATIVE:
  - C++ std::sort uses introsort (quicksort + heapsort) and sorts IN-PLACE
  - Our version allocates new lists at each level → more GC pressure
  - But: our version is trivially correct and has no mutation bugs
-}
mergeSort :: Ord a => [a] -> [a]
mergeSort []  = []        -- Base case: empty list is sorted
mergeSort [x] = [x]       -- Base case: single element is sorted
mergeSort xs  =
    let mid        = length xs `div` 2      -- Find midpoint
        (left, right) = splitAt mid xs      -- Split into two halves
        sortedLeft  = mergeSort left        -- Recursively sort left half
        sortedRight = mergeSort right       -- Recursively sort right half
    in merge sortedLeft sortedRight         -- Merge the sorted halves

{-|
  Merge two sorted lists into a single sorted list.

  HOW IT WORKS:
  - Compare the heads of both lists
  - Take the smaller one and prepend it to the result
  - Recurse on the remaining elements
  - When one list is empty, append the other (it's already sorted)

  This is where the actual "work" of sorting happens.
  Each call does O(1) work, and we make O(n) calls total → O(n) per merge level.

  KEY FP INSIGHT:
    This function is PURE — it takes two lists and returns a new list.
    No mutation anywhere. This purity is what makes parallelism safe:
    two threads can merge different sublists without any synchronization.
-}
merge :: Ord a => [a] -> [a] -> [a]
merge [] ys = ys                           -- Left exhausted → take right
merge xs [] = xs                           -- Right exhausted → take left
merge (x:xs) (y:ys)
    | x <= y    = x : merge xs (y:ys)     -- x is smaller → take x
    | otherwise = y : merge (x:xs) ys     -- y is smaller → take y
