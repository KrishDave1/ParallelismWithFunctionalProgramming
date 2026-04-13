{- Standalone benchmark for Word Count (MapReduce) -}
module Main where

import Bench.Utils
import MapReduce.SequentialWordCount (wordCount, topN)
import MapReduce.ParallelWordCount (parallelWordCount)
import Control.DeepSeq (force)
import Control.Exception (evaluate)

main :: IO ()
main = do
    printHeader "MapReduce Word Count Benchmark (Standalone)"
    let sampleText = unwords $ concatMap (\i ->
            [ "the", "quick", "brown", "fox", "jumps", "over"
            , "functional", "programming", "parallelism"
            , "haskell", "purity", "immutability"
            , "word" ++ show i
            ]) [1..10000 :: Int]
    
    let sizes = [1, 2, 5, 10]
    mapM_ (\mult -> do
        let text = concat $ replicate mult sampleText
        _ <- evaluate (force text)
        putStrLn $ "\n  Text: " ++ show (length $ words text) ++ " words"
        
        (seqR, seqT) <- timeIt (return $ wordCount text)
        printResult "  Sequential" seqT
        
        mapM_ (\c -> do
            (parR, parT) <- timeIt (return $ parallelWordCount c text)
            printResult ("  Parallel (" ++ show c ++ " chunks)") parT
            putStrLn $ "    Speedup: " ++ show (realToFrac seqT / realToFrac parT :: Double) ++ "x"
            putStrLn $ "    Correct: " ++ show (seqR == parR)
            ) [2, 4, 8]
        ) sizes
