{-
================================================================================
  MapReduce.SequentialWordCount — Sequential Word Count (Baseline)
================================================================================

PURPOSE:
  Count word frequencies in a text using pure functional programming.
  This baseline demonstrates how higher-order functions (map, fold, filter)
  naturally express the MapReduce pattern.

KEY FP CONCEPTS DEMONSTRATED:

  1. MAP-REDUCE PATTERN:
     The MapReduce paradigm is INHERENTLY FUNCTIONAL:
       - MAP phase: Apply a function to each element independently
       - REDUCE phase: Combine results using a fold (accumulation)
     
     Google's MapReduce framework (2004) was directly inspired by
     'map' and 'reduce' from functional programming languages (Lisp).
     
     Our word count follows this pattern:
       MAP:    text → split into words → map each word to (word, 1)
       REDUCE: group by word → fold each group by summing counts

  2. FUNCTION COMPOSITION (.) and PIPELINES:
     Haskell lets us compose functions with (.) operator:
       wordCount = reducePhase . mapPhase . tokenize
     
     This reads RIGHT to LEFT: tokenize, then mapPhase, then reducePhase.
     Each step is a pure function — no intermediate variables needed.
     
     Compare with imperative style:
       tokens = tokenize(text);
       mapped = mapPhase(tokens);
       result = reducePhase(mapped);
     Three mutable variables vs. one composed expression.

  3. HIGHER-ORDER FUNCTIONS:
     - 'map': transforms each element
     - 'foldr'/'foldl'': combines elements into a single result
     - 'filter': selects elements matching a predicate
     - 'groupBy': groups consecutive equal elements
     - 'sortBy': sorts by a comparison function
     
     These are the building blocks. By composing them, we build
     complex data processing pipelines without writing any loops.

  4. Data.Map AS AN EFFICIENT DICTIONARY:
     We use Haskell's 'Data.Map' (balanced binary tree) for counting.
     'insertWith (+) word 1 m' either inserts (word, 1) or increments
     the existing count. This is both pure AND efficient: O(log n).

================================================================================
-}

module MapReduce.SequentialWordCount
    ( wordCount
    , wordCountFromFile
    , tokenize
    , topN
    , WordFreq
    ) where

import qualified Data.Map.Strict as Map
import Data.Char (toLower, isAlpha)
import Control.DeepSeq (NFData(..))
import Data.List (sortBy)
import Data.Ord (Down(..))

-- | Type alias for word frequency maps
type WordFreq = Map.Map String Int

{-|
  Count word frequencies in a text string.

  PIPELINE:
    text
    → tokenize (split into lowercase words)
    → countWords (build frequency map)

  IMPLEMENTATION:
    We use 'foldl'' (strict left fold) to build the frequency map
    in a single pass through the word list. This is O(n log m) where
    n = total words and m = unique words.
    
    WHY STRICT FOLD (foldl' not foldl)?
    Haskell's lazy foldl builds up a chain of unevaluated thunks:
      foldl f z [1,2,3] = f (f (f z 1) 2) 3  -- thunks pile up!
    foldl' forces evaluation at each step, preventing stack overflow
    on large inputs. This is a common Haskell performance pattern.
-}
wordCount :: String -> WordFreq
wordCount = countWords . tokenize

{-|
  Read a file and count word frequencies.
-}
wordCountFromFile :: FilePath -> IO WordFreq
wordCountFromFile path = do
    contents <- readFile path
    return $! wordCount contents  -- $! forces evaluation (strict application)

{-|
  Tokenize text into lowercase words.

  STEPS:
    1. Convert to lowercase (map toLower)
    2. Replace non-alphabetic characters with spaces
    3. Split on whitespace into words
    4. Filter out empty strings

  KEY FP CONCEPT: This is a PIPELINE of transformations
    Each step is a pure function that transforms the data.
    No mutation, no intermediate state. Just data flowing through functions.
-}
tokenize :: String -> [String]
tokenize =
    words                           -- 4. Split on whitespace
    . map (\c -> if isAlpha c then toLower c else ' ')  -- 1-3. Normalize
                                    -- 'words' handles filtering empty strings

{-|
  Build a frequency map from a list of words.

  USES: Data.Map.Strict.insertWith for efficient accumulation.
    insertWith (+) word 1 map:
    - If 'word' is NOT in the map → insert (word, 1)
    - If 'word' IS in the map     → add 1 to existing count
  
  This is the REDUCE phase of MapReduce.
-}
countWords :: [String] -> WordFreq
countWords = foldl' (\acc w -> Map.insertWith (+) w 1 acc) Map.empty
  where
    -- Using strict foldl to avoid thunk accumulation
    foldl' _ z [] = z
    foldl' f z (x:xs) = let z' = f z x in z' `seq` foldl' f z' xs

{-|
  Get the top N most frequent words, sorted by frequency (descending).

  USES: sortBy with Down for descending order.
  This demonstrates Haskell's composability:
    toList → sort → take → done
-}
topN :: Int -> WordFreq -> [(String, Int)]
topN n = take n . sortBy (\(_, c1) (_, c2) -> compare (Down c1) (Down c2)) . Map.toList
