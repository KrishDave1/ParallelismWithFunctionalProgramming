{- Standalone benchmark for Matrix Multiplication -}
module Main where

import Bench.Utils
import Matrix.SequentialMatMul (matMul)
import Matrix.ParallelMatMul (parallelMatMul, parallelMatMulChunked)
import Control.DeepSeq (force)
import Control.Exception (evaluate)

main :: IO ()
main = do
    printHeader "Matrix Multiplication Benchmark (Standalone)"
    let sizes = [64, 128, 256, 512]
    mapM_ benchSize sizes

benchSize :: Int -> IO ()
benchSize n = do
    putStrLn $ "\n  Matrix size: " ++ show n ++ "x" ++ show n
    let a = generateRandomMatrix 42 n n 0 100
        b = generateRandomMatrix 99 n n 0 100
    _ <- evaluate (force a)
    _ <- evaluate (force b)
    
    (_, seqT) <- timeIt (return $ matMul a b)
    printResult "  Sequential" seqT
    
    (_, parT) <- timeIt (return $ parallelMatMul a b)
    printResult "  Parallel (parMap)" parT
    putStrLn $ "    Speedup: " ++ show (realToFrac seqT / realToFrac parT :: Double) ++ "x"
    
    mapM_ (\c -> do
        (_, chT) <- timeIt (return $ parallelMatMulChunked c a b)
        printResult ("  Chunked (" ++ show c ++ ")") chT
        putStrLn $ "    Speedup: " ++ show (realToFrac seqT / realToFrac chT :: Double) ++ "x"
        ) [2, 4, 8]
