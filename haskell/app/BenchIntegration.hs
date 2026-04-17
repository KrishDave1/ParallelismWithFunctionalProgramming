{- Standalone benchmark for Numerical Integration -}
module Main where

import Bench.Utils
import NumericalIntegration.SequentialIntegration
import NumericalIntegration.ParallelIntegration

main :: IO ()
main = do
    printHeader "Numerical Integration Benchmark (Numerical Simulation)"
    putStrLn "  FP Concepts: Higher-order functions, domain decomposition"
    putStrLn "  Parallelism: Domain decomposition + parallel evaluation"
    putStrLn ""
    
    let sizes = [1000000, 5000000, 10000000, 50000000]
    
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
                (parResult, parTime) <- timeIt (return $ parallelIntegrate (func tf) (lowerBound tf) (upperBound tf) n chunks)
                let speedup = realToFrac seqTime / realToFrac parTime :: Double
                printResult ("Parallel (" ++ show chunks ++ " chunks)") parTime
                putStrLn $ "    Result: " ++ show parResult
                putStrLn $ "    Speedup: " ++ show (roundTo 2 speedup) ++ "x"
                ) [2, 4, 8]
            
            putStrLn ""
            ) sizes
        ) testFunctions
  where
    roundTo :: Int -> Double -> Double
    roundTo n x = fromIntegral (round (x * 10^n) :: Int) / fromIntegral (10^n :: Int)
