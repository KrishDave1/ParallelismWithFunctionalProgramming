{- Standalone benchmark for Monte Carlo Pi -}
module Main where

import Bench.Utils
import MonteCarlo.SequentialPi (estimatePi)
import MonteCarlo.ParallelPi (parallelPiAsync, parallelPiSTM)

main :: IO ()
main = do
    printHeader "Monte Carlo Pi Estimation Benchmark (Standalone)"
    let sizes = [100000, 1000000, 10000000, 100000000]
    mapM_ benchSize sizes

benchSize :: Int -> IO ()
benchSize n = do
    putStrLn $ "\n  Samples: " ++ show n
    
    (seqR, seqT) <- timeIt (return $ estimatePi 42 n)
    printResult "  Sequential" seqT
    putStrLn $ "    π ≈ " ++ show seqR
    
    mapM_ (\w -> do
        (parR, parT) <- timeIt (parallelPiAsync w n 42)
        printResult ("  Async (" ++ show w ++ " workers)") parT
        putStrLn $ "    π ≈ " ++ show parR
        putStrLn $ "    Speedup: " ++ show (realToFrac seqT / realToFrac parT :: Double) ++ "x"
        ) [2, 4, 8]
    
    (stmR, stmT) <- timeIt (parallelPiSTM 4 n 42)
    printResult "  STM (4 workers)" stmT
    putStrLn $ "    π ≈ " ++ show stmR
    putStrLn $ "    Speedup: " ++ show (realToFrac seqT / realToFrac stmT :: Double) ++ "x"
