module Main where

import Sequential.QuickSort
import Parallel.ParQuickSort
import Sequential.MatrixMult
import Parallel.ParMatrixMult
import Parallel.Mandelbrot
import Utils.Timing

import Control.DeepSeq

main :: IO ()
main = do
    let list = [10000,9999..1] ::[Int]

    putStrLn "Sequential QuickSort:"
    timeIt $ print (take 10 (quicksort list))

    putStrLn "Parallel QuickSort:"
    timeIt $ print (take 10 (parQuicksort 0 50000 list))

    let m1 = replicate 800 (replicate 800 1)
    let m2 = replicate 800 (replicate 800 1)

    putStrLn "Matrix Multiplication:"
    timeIt $ print (head (matMult m1 m2))

    putStrLn "Parallel Matrix Multiplication:"
    timeIt $ print (take 1 (parMultiply m1 m2))

    putStrLn "Mandelbrot:"
    timeIt $ print (take 1 (mandelbrot 800 800 1000))