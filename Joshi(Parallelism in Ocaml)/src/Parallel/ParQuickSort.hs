module Parallel.ParQuickSort (parQuicksort) where

import Control.Parallel.Strategies
import Control.DeepSeq (NFData, force)

-- Sequential fallback
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
  quicksort smaller ++ [x] ++ quicksort bigger
  where
    smaller = [a | a <- xs, a <= x]
    bigger  = [a | a <- xs, a > x]

-- Parallel QuickSort with threshold + depth control
parQuicksort :: (NFData a, Ord a) => Int -> Int -> [a] -> [a]
parQuicksort _ _ [] = []
parQuicksort depth threshold xs@(x:rest)
  | length xs < threshold || depth > 3 = quicksort xs
  | otherwise =
      runEval $ do
        let smaller = [a | a <- rest, a <= x]
        let bigger  = [a | a <- rest, a > x]

        left  <- rpar (force (parQuicksort (depth + 1) threshold smaller))
        right <- rseq (parQuicksort (depth + 1) threshold bigger)

        return (left ++ [x] ++ right)