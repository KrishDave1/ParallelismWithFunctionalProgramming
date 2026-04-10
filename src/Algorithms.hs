module Algorithms (quicksort, matrixMultiply) where

-- | Sequential quicksort implementation
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = 
    let smallerSorted = quicksort [a | a <- xs, a <= x]
        biggerSorted  = quicksort [a | a <- xs, a > x]
    in  smallerSorted ++ [x] ++ biggerSorted

-- | Sequential matrix multiplication
-- Assumes matrices are represented as lists of lists (rows)
type Matrix a = [[a]]

-- Transpose a matrix to easily multiply rows by columns
transpose :: [[a]] -> [[a]]
transpose ([]:_) = []
transpose x = (map head x) : transpose (map tail x)

-- Calculate dot product of two lists
dotProduct :: Num a => [a] -> [a] -> a
dotProduct xs ys = sum $ zipWith (*) xs ys

-- Multiply two matrices
matrixMultiply :: Num a => Matrix a -> Matrix a -> Matrix a
matrixMultiply a b = 
    let bT = transpose b
    in [[dotProduct row col | col <- bT] | row <- a]
