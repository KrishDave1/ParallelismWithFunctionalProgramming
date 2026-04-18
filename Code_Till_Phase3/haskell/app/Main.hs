{-
================================================================================
  Main.hs — Master Benchmark Runner
================================================================================

PURPOSE:
  This is the main entry point that runs ALL four benchmarks sequentially
  and produces a comprehensive summary. Each benchmark:
    1. Runs the sequential version
    2. Runs the parallel version
    3. Computes speedup ratio
    4. Saves results to CSV for visualization

USAGE:
  cabal run haskell-parallelism -- +RTS -N4 -s
  
  FLAGS:
    +RTS        : begin runtime system options
    -N4         : use 4 OS threads (change to match your cores)
    -s          : print GC and spark statistics
    -N          : use all available cores (auto-detect)
================================================================================
-}

module Main where

import Bench.Utils
import Sorting.SequentialMergeSort (mergeSort)
import Sorting.ParallelMergeSort (parallelMergeSortWithDepth)
import Matrix.SequentialMatMul (matMul)
import Matrix.ParallelMatMul (parallelMatMul, parallelMatMulChunked)

import Control.DeepSeq (force)
import Control.Exception (evaluate)

main :: IO ()
main = do
    putStrLn ""
    putStrLn "╔══════════════════════════════════════════════════════════════════╗"
    putStrLn "║   Harnessing Functional Programming for Parallelism            ║"
    putStrLn "║   Comprehensive Benchmark Suite                                ║"
    putStrLn "╚══════════════════════════════════════════════════════════════════╝"
    
    benchMergeSort
    benchMatMul

    
    putStrLn ""
    putStrLn "All benchmarks complete! Results saved to results/ directory."

-- ============================================================================
-- Benchmark 1: Parallel Merge Sort
-- ============================================================================

benchMergeSort :: IO ()
benchMergeSort = do
    printHeader "BENCHMARK 1: Parallel Merge Sort"
    putStrLn "  FP Concepts: Eval Monad, rpar/rseq, Strategies, Depth Threshold"
    putStrLn "  Parallelism: Task Parallelism (divide-and-conquer)"
    putStrLn ""
    
    -- Test with different input sizes
    let sizes = [10000, 50000, 100000, 500000]
    
    mapM_ (\size -> do
        putStrLn $ "  --- Input size: " ++ show size ++ " elements ---"
        
        -- Generate deterministic random data
        let xs = generateRandomList 42 size 1 (size * 10)
        -- Force evaluation of input data (don't include generation in timing)
        _ <- evaluate (force xs)
        
        -- Sequential baseline
        (seqResult, seqTime) <- timeIt (return $ mergeSort xs)
        printResult "Sequential Merge Sort" seqTime
        
        -- Parallel with different depth thresholds
        (_parResult2, parTime2) <- timeIt (return $ parallelMergeSortWithDepth 2 xs)
        let speedup2 = realToFrac seqTime / realToFrac parTime2 :: Double
        printResult ("Parallel (depth=2, 4 tasks)") parTime2
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup2) ++ "x"
        
        (_parResult3, parTime3) <- timeIt (return $ parallelMergeSortWithDepth 3 xs)
        let speedup3 = realToFrac seqTime / realToFrac parTime3 :: Double
        printResult ("Parallel (depth=3, 8 tasks)") parTime3
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup3) ++ "x"
        
        (parResult4, parTime4) <- timeIt (return $ parallelMergeSortWithDepth 4 xs)
        let speedup4 = realToFrac seqTime / realToFrac parTime4 :: Double
        printResult ("Parallel (depth=4, 16 tasks)") parTime4
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup4) ++ "x"
        
        -- Verify correctness: parallel result should equal sequential
        let correct = seqResult == parResult4
        putStrLn $ "    Correctness check: " ++ (if correct then "PASS" else "FAIL")
        putStrLn ""
        ) sizes

-- ============================================================================
-- Benchmark 2: Parallel Matrix Multiplication
-- ============================================================================

benchMatMul :: IO ()
benchMatMul = do
    printHeader "BENCHMARK 2: Parallel Matrix Multiplication"
    putStrLn "  FP Concepts: Data Parallelism, parMap, Strategies, Chunking"
    putStrLn "  Parallelism: Data Parallelism (same operation on different data)"
    putStrLn ""
    
    let sizes = [64, 128, 256]
    
    mapM_ (\size -> do
        putStrLn $ "  --- Matrix size: " ++ show size ++ "x" ++ show size ++ " ---"
        
        let matA = generateRandomMatrix 42 size size 0 100
            matB = generateRandomMatrix 99 size size 0 100
        _ <- evaluate (force matA)
        _ <- evaluate (force matB)
        
        -- Sequential
        (_seqResult, seqTime) <- timeIt (return $ matMul matA matB)
        printResult "Sequential Matrix Multiply" seqTime
        
        -- Parallel (parMap)
        (_parResult, parTime) <- timeIt (return $ parallelMatMul matA matB)
        let speedup = realToFrac seqTime / realToFrac parTime :: Double
        printResult "Parallel (parMap over rows)" parTime
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
        
        -- Parallel (chunked, 4 chunks)
        (_chunkResult, chunkTime) <- timeIt (return $ parallelMatMulChunked 4 matA matB)
        let speedupC = realToFrac seqTime / realToFrac chunkTime :: Double
        printResult "Parallel (chunked, 4 chunks)" chunkTime
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedupC) ++ "x"
        
        -- Parallel (chunked, 8 chunks)
        (_, chunkTime8) <- timeIt (return $ parallelMatMulChunked 8 matA matB)
        let speedupC8 = realToFrac seqTime / realToFrac chunkTime8 :: Double
        printResult "Parallel (chunked, 8 chunks)" chunkTime8
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedupC8) ++ "x"
        
        putStrLn ""
        ) sizes



roundTo :: Int -> Double -> Double
roundTo n x = fromIntegral (round (x * 10^n) :: Int) / fromIntegral (10^n :: Int)
