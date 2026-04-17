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

import KMeans.SequentialKMeans
import KMeans.ParallelKMeans
import System.Random (mkStdGen, uniformR)

import NumericalIntegration.SequentialIntegration
import NumericalIntegration.ParallelIntegration

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
    benchWordCount
    benchMonteCarlo
    benchKMeans
    benchIntegration
    
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
    
    let textSizes = [(1 :: Int, sampleText), (3, concat $ replicate 3 sampleText)]
    
    mapM_ (\(_mult, text) -> do
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

-- ============================================================================
-- Benchmark 5: K-Means Clustering
-- ============================================================================

-- | Generate random 2D points clustered around K centers
generateClusteredData :: Int -> Int -> Int -> [Point]
generateClusteredData seed k totalPoints =
    let pointsPerCluster = totalPoints `div` k
        -- Generate K cluster centers evenly spaced
        centers = [(100 * cos (2 * pi * fromIntegral i / fromIntegral k),
                    100 * sin (2 * pi * fromIntegral i / fromIntegral k))
                  | i <- [0..k-1]]
    in concatMap (\(cx, cy) -> generateAround seed cx cy pointsPerCluster) centers
  where
    generateAround s cx cy n =
        let go _ [] = []
            go gen (_:rest) =
                let (dx, gen')  = uniformR (-20.0, 20.0 :: Double) gen
                    (dy, gen'') = uniformR (-20.0, 20.0 :: Double) gen'
                in (cx + dx, cy + dy) : go gen'' rest
        in go (mkStdGen (s + round cx + round cy)) [1..n]

benchKMeans :: IO ()
benchKMeans = do
    printHeader "BENCHMARK 5: K-Means Clustering (Machine Learning)"
    putStrLn "  FP Concepts: Immutable state, higher-order map, Strategies"
    putStrLn "  Parallelism: Data Parallelism (parallel assignment step)"
    putStrLn ""
    
    let k = 5  -- number of clusters
    
    mapM_ (\numPoints -> do
        putStrLn $ "  --- " ++ show numPoints ++ " points, K=" ++ show k ++ " ---"
        
        let points = generateClusteredData 42 k numPoints
        _ <- evaluate (force points)
        
        -- Use first K points as initial centroids (simple initialization)
        let initCentroids = take k points
        
        -- Sequential
        (seqResult, seqTime) <- timeIt (return $ kMeans 100 0.001 initCentroids points)
        printResult "Sequential K-Means" seqTime
        let round' x = fromIntegral (round (x * 100) :: Int) / 100.0 :: Double
        putStrLn $ "    Final centroids: " ++ show (map (\(x,y) -> (round' x, round' y)) seqResult)
        
        -- Parallel with different chunk sizes
        mapM_ (\chunks -> do
            let chunkSize = max 1 (numPoints `div` chunks)
            (_parResult, parTime) <- timeIt (return $ parallelKMeans chunkSize 100 0.001 initCentroids points)
            let speedup = realToFrac seqTime / realToFrac parTime :: Double
            printResult ("Parallel (" ++ show chunks ++ " chunks)") parTime
            putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
            ) [2, 4, 8]
        
        putStrLn ""
        ) [10000, 50000, 100000]

-- ============================================================================
-- Benchmark 6: Numerical Integration
-- ============================================================================

benchIntegration :: IO ()
benchIntegration = do
    printHeader "BENCHMARK 6: Numerical Integration (Numerical Simulation)"
    putStrLn "  FP Concepts: Higher-order functions, domain decomposition"
    putStrLn "  Parallelism: Domain decomposition + parallel evaluation"
    putStrLn ""
    
    let sizes = [1000000, 5000000, 10000000]
    
    mapM_ (\tf -> do
        putStrLn $ "  === Function: " ++ funcName tf ++ " ==="
        putStrLn $ "  Exact answer: " ++ show (exactAnswer tf)
        putStrLn ""
        
        mapM_ (\n -> do
            putStrLn $ "  --- N = " ++ show n ++ " sub-intervals ---"
            
            -- Sequential (strict)
            (seqResult, seqTime) <- timeIt (return $ integrateStrict (func tf) (lowerBound tf) (upperBound tf) n)
            printResult "Sequential (strict)" seqTime
            putStrLn $ "    Result: " ++ show seqResult
            putStrLn $ "    Error:  " ++ show (abs (seqResult - exactAnswer tf))
            
            -- Parallel (domain decomposition) with different chunk counts
            mapM_ (\chunks -> do
                (_parResult, parTime) <- timeIt (return $ parallelIntegrate (func tf) (lowerBound tf) (upperBound tf) n chunks)
                let speedup = realToFrac seqTime / realToFrac parTime :: Double
                printResult ("Parallel (" ++ show chunks ++ " chunks)") parTime
                putStrLn $ "    Result: " ++ show _parResult
                putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
                ) [2, 4, 8]
            
            putStrLn ""
            ) sizes
        ) testFunctions

roundTo :: Int -> Double -> Double
roundTo n x = fromIntegral (round (x * 10^n) :: Int) / fromIntegral (10^n :: Int)
