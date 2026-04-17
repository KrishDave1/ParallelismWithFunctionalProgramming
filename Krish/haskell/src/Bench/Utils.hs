{-
================================================================================
  Bench.Utils — Benchmarking Utilities
================================================================================

PURPOSE:
  This module provides common utilities for benchmarking our parallel algorithms.
  It includes functions for:
    - Measuring wall-clock execution time
    - Generating random test data
    - Writing benchmark results to CSV files
    - Pretty-printing results to the console

DESIGN DECISIONS:
  - We use Data.Time for wall-clock timing (not CPU time) because we are
    measuring parallelism speedup, which is about real elapsed time.
  - DeepSeq (NFData) is used to force full evaluation of results before
    stopping the timer — this prevents lazy evaluation from skewing measurements.
  - Results are written to CSV for later visualization with Python matplotlib.

KEY FP CONCEPTS USED:
  - Higher-order functions: `benchmark` takes a function as argument
  - Polymorphism: works with any type that has NFData instance
  - Pure functions for data generation (deterministic via seed)
================================================================================
-}

module Bench.Utils
    ( -- * Timing
      timeIt
    , timeItN
      -- * Data Generation
    , generateRandomList
    , generateRandomMatrix
      -- * CSV Output
    , writeCSV
    , appendCSV
      -- * Display
    , printHeader
    , printResult
    , formatTime
    ) where

import Control.DeepSeq (NFData, force)
import Control.Exception (evaluate)
import Data.Time.Clock (getCurrentTime, diffUTCTime, NominalDiffTime)
import System.Random (mkStdGen, uniformR, StdGen)
import System.Directory (createDirectoryIfMissing)
import System.IO (withFile, IOMode(..), hPutStrLn)

-- ============================================================================
-- Timing Utilities
-- ============================================================================

{-|
  @timeIt@ measures the wall-clock time of executing a computation.

  HOW IT WORKS:
    1. Record the start time
    2. Force the computation to normal form (fully evaluate it)
       using 'evaluate . force' — this is critical because Haskell
       is lazy, and without forcing, we'd just measure thunk creation
    3. Record the end time
    4. Return both the result and elapsed time

  WHY 'evaluate . force'?
    - 'force' (from DeepSeq) recursively evaluates the entire data structure
      to Normal Form (NF), not just Weak Head Normal Form (WHNF)
    - Without this, measuring "sort [5,3,1,4,2]" would only force the
      first cons cell, not the entire sorted list
    - 'evaluate' ensures the forcing happens in IO (not just building a thunk)

  EXAMPLE:
    (result, elapsed) <- timeIt (return $ sort bigList)
-}
timeIt :: NFData a => IO a -> IO (a, NominalDiffTime)
timeIt action = do
    start  <- getCurrentTime
    result <- action >>= evaluate . force
    end    <- getCurrentTime
    let elapsed = diffUTCTime end start
    return (result, elapsed)

{-|
  @timeItN@ runs a computation @n@ times and returns the median time.
  This reduces measurement noise from OS scheduling, GC pauses, etc.

  We return the median (not mean) because it's more robust to outliers.
-}
timeItN :: NFData a => Int -> IO a -> IO (a, NominalDiffTime)
timeItN n action = do
    times <- mapM (\_ -> snd <$> timeIt action) [1 .. n - 1]
    (result, lastTime) <- timeIt action
    let allTimes  = lastTime : times
        sorted    = sortTimes allTimes
        medianIdx = length sorted `div` 2
        median    = sorted !! medianIdx
    return (result, median)
  where
    sortTimes []  = []
    sortTimes (x:xs) = sortTimes [y | y <- xs, y <= x]
                    ++ [x]
                    ++ sortTimes [y | y <- xs, y > x]

-- ============================================================================
-- Data Generation
-- ============================================================================

{-|
  Generate a list of random integers using a deterministic seed.

  KEY FP CONCEPT: Pure data generation
    By using a fixed seed, we get the exact same list every time.
    This is essential for fair benchmarking — all algorithms sort
    the exact same data.

  PARAMETERS:
    - seed: deterministic seed for reproducibility
    - n: number of elements to generate
    - lo, hi: range of random values [lo, hi]
-}
generateRandomList :: Int -> Int -> Int -> Int -> [Int]
generateRandomList seed n lo hi =
    take n $ unfoldr' (mkStdGen seed)
  where
    unfoldr' :: StdGen -> [Int]
    unfoldr' gen =
        let (val, gen') = uniformR (lo, hi) gen
        in val : unfoldr' gen'

{-|
  Generate a random matrix (list of lists) for matrix multiplication benchmarks.

  PARAMETERS:
    - seed: deterministic seed
    - rows, cols: dimensions
    - lo, hi: value range
-}
generateRandomMatrix :: Int -> Int -> Int -> Int -> Int -> [[Double]]
generateRandomMatrix seed rows cols lo hi =
    chunksOf cols allValues
  where
    allValues = take (rows * cols) $ unfoldr' (mkStdGen seed)
    unfoldr' :: StdGen -> [Double]
    unfoldr' gen =
        let (val, gen') = uniformR (fromIntegral lo, fromIntegral hi) gen
        in val : unfoldr' gen'
    chunksOf :: Int -> [a] -> [[a]]
    chunksOf _ [] = []
    chunksOf k xs = let (a, b) = splitAt k xs in a : chunksOf k b

-- ============================================================================
-- CSV Output (for Python visualization)
-- ============================================================================

{-|
  Write benchmark results to a CSV file. Creates the directory if needed.

  CSV FORMAT:
    problem,language,variant,input_size,num_cores,time_seconds,speedup

  WHY CSV?
    - Universal format readable by pandas, Excel, matplotlib
    - Easy to append results from C++ and Python benchmarks
    - Enables automated chart generation
-}
writeCSV :: FilePath -> [String] -> [[String]] -> IO ()
writeCSV filepath headers rows = do
    createDirectoryIfMissing True (takeDirectory filepath)
    withFile filepath WriteMode $ \h -> do
        hPutStrLn h (intercalate "," headers)
        mapM_ (hPutStrLn h . intercalate ",") rows
  where
    takeDirectory = reverse . dropWhile (/= '/') . reverse
    intercalate sep = foldr1 (\a b -> a ++ sep ++ b)

{-|
  Append a single row to an existing CSV file.
-}
appendCSV :: FilePath -> [String] -> IO ()
appendCSV filepath row =
    withFile filepath AppendMode $ \h ->
        hPutStrLn h (intercalate "," row)
  where
    intercalate sep = foldr1 (\a b -> a ++ sep ++ b)

-- ============================================================================
-- Display Utilities
-- ============================================================================

{-|
  Print a formatted section header for benchmark output.
-}
printHeader :: String -> IO ()
printHeader title = do
    putStrLn ""
    putStrLn $ replicate 70 '='
    putStrLn $ "  " ++ title
    putStrLn $ replicate 70 '='
    putStrLn ""

{-|
  Print a single benchmark result in a readable format.
-}
printResult :: String -> NominalDiffTime -> IO ()
printResult label elapsed =
    putStrLn $ "  " ++ padRight 40 label ++ formatTime elapsed
  where
    padRight n s = s ++ replicate (max 0 (n - length s)) ' '

{-|
  Format a NominalDiffTime as a human-readable string.
  Shows milliseconds for short durations, seconds for longer ones.
-}
formatTime :: NominalDiffTime -> String
formatTime t
    | seconds < 0.001 = show (seconds * 1000000) ++ " μs"
    | seconds < 1.0   = show (roundTo 2 (seconds * 1000)) ++ " ms"
    | otherwise       = show (roundTo 3 seconds) ++ " s"
  where
    seconds = realToFrac t :: Double
    roundTo :: Int -> Double -> Double
    roundTo n x = fromIntegral (round (x * 10^n) :: Int) / 10^(fromIntegral n)
