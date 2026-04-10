module Main where

import Algorithms

main :: IO ()
main = do
    putStrLn "--- Function Parallelism: Week 1 ---"
    putStrLn "Sequential Quicksort:"
    let unsortedList = [5, 2, 8, 1, 9, 3, 7, 4, 6] :: [Int]
    putStrLn $ "Unsorted: " ++ show unsortedList
    putStrLn $ "Sorted:   " ++ show (quicksort unsortedList)

    putStrLn "\nSequential Matrix Multiplication:"
    let matA = [[1, 2], [3, 4]] :: [[Int]]
    let matB = [[2, 0], [1, 2]] :: [[Int]]
    putStrLn $ "Mat A: " ++ show matA
    putStrLn $ "Mat B: " ++ show matB
    putStrLn $ "A * B: " ++ show (matrixMultiply matA matB)
