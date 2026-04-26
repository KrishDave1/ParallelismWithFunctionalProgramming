module Parallel.ParMatrixMult (parMultiply) where

import Control.Parallel.Strategies hiding (dot)

type Matrix = [[Int]]

-- Dot product of two vectors
dot :: [Int] -> [Int] -> Int
dot xs ys = sum (zipWith (*) xs ys)

-- Transpose matrix
transpose :: Matrix -> Matrix
transpose ([]:_) = []
transpose x = map head x : transpose (map tail x)

-- Parallel matrix multiplication
parMultiply :: Matrix -> Matrix -> Matrix
parMultiply a b =
  let bt = transpose b
  in parMap rdeepseq (\row ->
        map (dot row) bt
     ) a