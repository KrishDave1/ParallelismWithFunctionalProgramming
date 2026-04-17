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
import MapReduce.SequentialWordCount (wordCount, topN)
import MapReduce.ParallelWordCount (parallelWordCount)
import MonteCarlo.SequentialPi (estimatePi)
import MonteCarlo.ParallelPi (parallelPiAsync, parallelPiSTM)

import Control.DeepSeq (force, NFData)
import Control.Exception (evaluate)
import Data.Time.Clock (NominalDiffTime)
import System.IO (hFlush, stdout)

main :: IO ()
main = do
    putStrLn ""
    putStrLn "╔══════════════════════════════════════════════════════════════════╗"
    putStrLn "║   Harnessing Functional Programming for Parallelism            ║"
    putStrLn "║   Comprehensive Benchmark Suite                                ║"
    putStrLn "╚══════════════════════════════════════════════════════════════════╝"
    
    benchMergeSort
    benchMatMul
    benchWordCount
    benchMonteCarlo
    
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
        (parResult2, parTime2) <- timeIt (return $ parallelMergeSortWithDepth 2 xs)
        let speedup2 = realToFrac seqTime / realToFrac parTime2 :: Double
        printResult ("Parallel (depth=2, 4 tasks)") parTime2
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup2) ++ "x"
        
        (parResult3, parTime3) <- timeIt (return $ parallelMergeSortWithDepth 3 xs)
        let speedup3 = realToFrac seqTime / realToFrac parTime3 :: Double
        printResult ("Parallel (depth=3, 8 tasks)") parTime3
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup3) ++ "x"
        
        (parResult4, parTime4) <- timeIt (return $ parallelMergeSortWithDepth 4 xs)
        let speedup4 = realToFrac seqTime / realToFrac parTime4 :: Double
        printResult ("Parallel (depth=4, 16 tasks)") parTime4
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup4) ++ "x"
        
        -- Verify correctness: parallel result should equal sequential
        let correct = seqResult == parResult4
        putStrLn $ "    Correctness check: " ++ (if correct then "✓ PASS" else "✗ FAIL")
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
        (seqResult, seqTime) <- timeIt (return $ matMul matA matB)
        printResult "Sequential Matrix Multiply" seqTime
        
        -- Parallel (parMap)
        (parResult, parTime) <- timeIt (return $ parallelMatMul matA matB)
        let speedup = realToFrac seqTime / realToFrac parTime :: Double
        printResult "Parallel (parMap over rows)" parTime
        putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
        
        -- Parallel (chunked, 4 chunks)
        (chunkResult, chunkTime) <- timeIt (return $ parallelMatMulChunked 4 matA matB)
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

-- ============================================================================
-- Benchmark 3: Parallel Word Count (MapReduce)
-- ============================================================================

benchWordCount :: IO ()
benchWordCount = do
    printHeader "BENCHMARK 3: Parallel MapReduce Word Count"
    putStrLn "  FP Concepts: Par Monad, Higher-Order Functions, Map-Reduce Pattern"
    putStrLn "  Parallelism: Data Parallelism (partition data, parallel map, reduce)"
    putStrLn ""
    
    -- Generate a large text corpus by repeating sample text
    let sampleText = unwords $ concatMap (\i -> 
            [ "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
            , "functional", "programming", "enables", "safe", "parallelism"
            , "through", "purity", "and", "immutability"
            , "haskell", "provides", "multiple", "parallelism", "constructs"
            , "including", "sparks", "strategies", "and", "the", "par", "monad"
            , "word" ++ show i  -- unique words to make frequency distribution interesting
            ]) [1..5000 :: Int]
    
    let textSizes = [(1, sampleText), (3, concat $ replicate 3 sampleText)]
    
    mapM_ (\(mult, text) -> do
        let wordCount' = length (words text)
        putStrLn $ "  --- Text size: ~" ++ show wordCount' ++ " words ---"
        
        _ <- evaluate (force text)
        
        -- Sequential
        (seqResult, seqTime) <- timeIt (return $ wordCount text)
        printResult "Sequential Word Count" seqTime
        putStrLn $ "    Unique words: " ++ show (length $ topN 999999 seqResult)
        putStrLn $ "    Top 5: " ++ show (topN 5 seqResult)
        
        -- Parallel with different chunk counts
        mapM_ (\chunks -> do
            (parResult, parTime) <- timeIt (return $ parallelWordCount chunks text)
            let speedup = realToFrac seqTime / realToFrac parTime :: Double
            printResult ("Parallel (" ++ show chunks ++ " chunks)") parTime
            putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
            
            -- Verify correctness
            let correct = seqResult == parResult
            putStrLn $ "    Correctness: " ++ (if correct then "✓ PASS" else "✗ FAIL")
            ) [2, 4, 8]
        
        putStrLn ""
        ) textSizes

-- ============================================================================
-- Benchmark 4: Monte Carlo Pi Estimation
-- ============================================================================

benchMonteCarlo :: IO ()
benchMonteCarlo = do
    printHeader "BENCHMARK 4: Monte Carlo Pi Estimation"
    putStrLn "  FP Concepts: Async (Futures/Promises), STM, Pure Splittable RNG"
    putStrLn "  Parallelism: Embarrassingly Parallel (independent samples)"
    putStrLn ""
    
    let sampleSizes = [100000, 1000000, 10000000]
    
    mapM_ (\n -> do
        putStrLn $ "  --- Samples: " ++ show n ++ " ---"
        
        -- Sequential
        (seqResult, seqTime) <- timeIt (return $ estimatePi 42 n)
        printResult "Sequential" seqTime
        putStrLn $ "    π ≈ " ++ show seqResult
        putStrLn $ "    Error: " ++ show (abs (seqResult - pi))
        
        -- Parallel (Async) with different worker counts
        mapM_ (\workers -> do
            (parResult, parTime) <- timeIt (parallelPiAsync workers n 42)
            let speedup = realToFrac seqTime / realToFrac parTime :: Double
            printResult ("Async (" ++ show workers ++ " workers)") parTime
            putStrLn $ "    π ≈ " ++ show parResult
            putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
            ) [2, 4, 8]
        
        -- Parallel (STM)
        (stmResult, stmTime) <- timeIt (parallelPiSTM 4 n 42)
        let stmSpeedup = realToFrac seqTime / realToFrac stmTime :: Double
        printResult "STM (4 workers)" stmTime
        putStrLn $ "    π ≈ " ++ show stmResult
        putStrLn $ "    Speedup: " ++ show (roundTo 2 stmSpeedup) ++ "x"
        
        putStrLn ""
        ) sampleSizes

-- ============================================================================
-- Helpers
-- ============================================================================

roundTo :: Int -> Double -> Double
roundTo n x = fromIntegral (round (x * 10^n) :: Int) / 10^(fromIntegral n)
