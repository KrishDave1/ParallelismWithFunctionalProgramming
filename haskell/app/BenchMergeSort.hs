{- Standalone benchmark for Merge Sort -}
module Main where

import Bench.Utils
import Sorting.SequentialMergeSort (mergeSort)
import Sorting.ParallelMergeSort (parallelMergeSortWithDepth)
import Control.DeepSeq (force)
import Control.Exception (evaluate)

main :: IO ()
main = do
    printHeader "Merge Sort Benchmark (Standalone)"
    let sizes = [10000, 50000, 100000, 200000, 500000, 1000000]
    mapM_ benchSize sizes

benchSize :: Int -> IO ()
benchSize n = do
    putStrLn $ "\n  Input size: " ++ show n
    let xs = generateRandomList 42 n 1 (n * 10)
    _ <- evaluate (force xs)
    
    (_, seqT) <- timeIt (return $ mergeSort xs)
    printResult "  Sequential" seqT
    
    mapM_ (\d -> do
        (_, parT) <- timeIt (return $ parallelMergeSortWithDepth d xs)
        let s = realToFrac seqT / realToFrac parT :: Double
        printResult ("  Parallel depth=" ++ show d) parT
        putStrLn $ "    Speedup: " ++ show s ++ "x"
        ) [1, 2, 3, 4, 5]
