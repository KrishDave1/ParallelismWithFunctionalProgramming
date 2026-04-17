{-
================================================================================
  MapReduce.ParallelWordCount — Parallel Word Count using Par Monad
================================================================================

PURPOSE:
  Parallel word frequency counting using Haskell's Par monad.
  This demonstrates a DIFFERENT parallelism approach from the Eval monad
  used in merge sort, showing how Haskell offers multiple parallelism
  abstractions at different levels of control.

KEY FP CONCEPTS DEMONSTRATED:

  1. THE Par MONAD (Control.Monad.Par):
     Unlike the Eval monad (which creates "sparks" — hints to the runtime),
     the Par monad provides DETERMINISTIC PARALLELISM with explicit
     communication channels.
     
     KEY PRIMITIVES:
       - 'fork': spawn a parallel computation (guaranteed to run)
       - 'new':  create a new IVar (write-once variable)
       - 'put':  write a value to an IVar (can only be done ONCE)
       - 'get':  read from an IVar (blocks until a value is available)
     
     COMPARISON WITH Eval MONAD:
       Eval: Spark-based, best-effort parallelism. Sparks may fizzle.
       Par:  Fork-based, guaranteed parallelism. Forks always execute.
     
     COMPARISON WITH IMPERATIVE THREADS:
       The Par monad is DETERMINISTIC — running the same computation
       always produces the same result, regardless of scheduling order.
       This is because IVars can only be written ONCE (no races).
       
       In contrast, threads with shared mutable state are NONDETERMINISTIC
       — different runs can produce different results due to scheduling.

  2. PARALLEL MAP-REDUCE:
     Our parallel strategy:
       a. SPLIT: Divide the text into N chunks (one per core)
       b. MAP:   Each chunk's words are counted independently (in parallel)
       c. REDUCE: Merge all frequency maps using Map.unionWith (+)
     
     This directly mirrors Google's MapReduce framework, but expressed
     as pure functional operations — no distributed systems needed.

  3. DATA PARTITIONING:
     We split the input text into roughly equal-sized chunks.
     Each chunk is processed by one parallel task.
     
     KEY INSIGHT: Because word counting is ASSOCIATIVE and COMMUTATIVE,
     we can split and recombine in any order without affecting results.
     This algebraic property is what makes MapReduce work — and it's
     naturally expressed in FP through monoids/semigroups.
     
     (String → WordFreq) is a monoid homomorphism:
       count(A ++ B) = count(A) `union(+)` count(B)

  4. HIGHER-ORDER FUNCTION 'parMapM':
     We use parMapM to apply our counting function in parallel.
     This is a MONADIC parallel map — it maps a function that lives
     in the Par monad over a list, executing all applications in parallel.

ARCHITECTURE OF OUR PARALLEL WORD COUNT:

    ┌──────────┐
    │Raw Text  │
    └────┬─────┘
         │ split into N chunks
    ┌────┴─────────────────────────┐
    │  ┌───────┐ ┌───────┐ ┌───────┐
    │  │Chunk 1│ │Chunk 2│ │Chunk N│    ← DATA PARALLELISM
    │  └───┬───┘ └───┬───┘ └───┬───┘
    │      │         │         │
    │  ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
    │  │Count 1│ │Count 2│ │Count N│    ← MAP PHASE (parallel)
    │  └───┬───┘ └───┴───┘ └───┬───┘
    │      │         │         │
    └──────┴────┬────┴─────────┘
                │ merge with unionWith (+)
         ┌──────┴──────┐
         │ Final Count │                  ← REDUCE PHASE
         └─────────────┘

================================================================================
-}

module MapReduce.ParallelWordCount
    ( parallelWordCount
    , parallelWordCountFromFile
    ) where

import MapReduce.SequentialWordCount (tokenize, WordFreq)
import qualified Data.Map.Strict as Map
import Control.Monad.Par (runPar, parMapM)

{-|
  Parallel word count using the Par monad.

  PARAMETERS:
    - numChunks: number of parallel tasks (ideally = number of cores)
    - text: the input text to process

  IMPLEMENTATION WALKTHROUGH:
    1. TOKENIZE the text into words (sequential — I/O-bound, not worth parallelizing)
    2. SPLIT the word list into 'numChunks' roughly equal chunks
    3. MAP PHASE: Count words in each chunk in parallel using parMapM
    4. REDUCE PHASE: Merge all chunk results with Map.unionWith (+)

  WHY THE Par MONAD HERE (not Eval)?
    - Word counting is compute-intensive per chunk → want guaranteed execution
    - Eval sparks might fizzle if the runtime decides the work is too small
    - Par monad guarantees all forks will execute
    - Par monad is deterministic → easier to test and debug

  PERFORMANCE CHARACTERISTICS:
    - Splitting: O(n) — one pass through the word list
    - Map phase: O(n/p) per core where n=words, p=numChunks
    - Reduce phase: O(m × p) where m=unique words (merging p maps)
    - Overall: O(n/p + m×p) — good speedup when n >> m×p
-}
parallelWordCount :: Int -> String -> WordFreq
parallelWordCount numChunks text =
    let allWords = tokenize text
        chunks   = splitIntoChunks numChunks allWords
    in runPar $ do
        -- MAP PHASE: count words in each chunk in parallel
        -- parMapM applies 'countChunk' to each chunk, running them
        -- in parallel using the Par monad's fork/join mechanism
        chunkResults <- parMapM (return . countChunk) chunks
        -- REDUCE PHASE: merge all frequency maps
        -- Map.unionWith (+) combines maps by adding counts for shared keys
        -- This is the "reduce" step of MapReduce
        return $ foldl1 (Map.unionWith (+)) chunkResults

{-|
  Count words in a single chunk (used within each parallel task).
  This is simply building a frequency map — the same as sequential,
  but applied to a smaller subset of the data.
-}
countChunk :: [String] -> WordFreq
countChunk = foldl' (\acc w -> Map.insertWith (+) w 1 acc) Map.empty
  where
    foldl' _ z [] = z
    foldl' f z (x:xs) = let z' = f z x in z' `seq` foldl' f z' xs

{-|
  Parallel word count from a file.
-}
parallelWordCountFromFile :: Int -> FilePath -> IO WordFreq
parallelWordCountFromFile numChunks path = do
    contents <- readFile path
    return $! parallelWordCount numChunks contents

-- ============================================================================
-- Helper: Split a list into N roughly equal chunks
-- ============================================================================

{-|
  Split a list into n chunks of roughly equal size.
  
  EXAMPLE:
    splitIntoChunks 3 [1..10] = [[1,2,3,4], [5,6,7], [8,9,10]]
  
  The first few chunks may be one element larger than the rest
  to handle cases where the list length is not evenly divisible.
-}
splitIntoChunks :: Int -> [a] -> [[a]]
splitIntoChunks n xs
    | n <= 1    = [xs]
    | null xs   = []
    | otherwise =
        let len       = length xs
            chunkSize = len `div` n
            remainder = len `mod` n
        in go chunkSize remainder xs
  where
    go _ _ [] = []
    go size rem' ys =
        let thisSize = size + (if rem' > 0 then 1 else 0)
            (chunk, rest) = splitAt thisSize ys
        in chunk : go size (max 0 (rem' - 1)) rest
