module Sequential.MatrixMult (matMult) where

type Matrix = [[Int]]

transpose :: Matrix -> Matrix
transpose ([]:_) = []
transpose x = map head x : transpose (map tail x)

dot :: [Int] -> [Int] -> Int
dot xs ys = sum (zipWith (*) xs ys)

matMult :: Matrix -> Matrix -> Matrix
matMult a b =
    let bT = transpose b
    in [[ dot row col | col <- bT ] | row <- a]