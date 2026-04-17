{-# LANGUAGE BangPatterns #-}
{- Standalone benchmark for K-Means Clustering -}
module Main where

import Bench.Utils
import KMeans.SequentialKMeans
import KMeans.ParallelKMeans
import Control.DeepSeq (force)
import Control.Exception (evaluate)
import System.Random (mkStdGen, uniformR)

-- | Generate random 2D points clustered around K centers
generateClusteredData :: Int -> Int -> Int -> [Point]
generateClusteredData seed k totalPoints =
    let pointsPerCluster = totalPoints `div` k
        gen0 = mkStdGen seed
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

main :: IO ()
main = do
    printHeader "K-Means Clustering Benchmark (Machine Learning)"
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
        putStrLn $ "    Final centroids: " ++ show (map (\(x,y) -> (round' x, round' y)) seqResult)
        
        -- Parallel with different chunk sizes
        mapM_ (\chunks -> do
            let chunkSize = max 1 (numPoints `div` chunks)
            (parResult, parTime) <- timeIt (return $ parallelKMeans chunkSize 100 0.001 initCentroids points)
            let (parCentroids, parIters) = parResult
            let speedup = realToFrac seqTime / realToFrac parTime :: Double
            printResult ("Parallel (" ++ show chunks ++ " chunks)") parTime
            putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
            putStrLn $ "    Iterations: " ++ show parIters
            ) [2, 4, 8]
        
        putStrLn ""
        ) [10000, 50000, 100000, 200000]
  where
    round' x = fromIntegral (round (x * 100) :: Int) / 100.0 :: Double
    roundTo :: Int -> Double -> Double
    roundTo n x = fromIntegral (round (x * 10^n) :: Int) / fromIntegral (10^n :: Int)
