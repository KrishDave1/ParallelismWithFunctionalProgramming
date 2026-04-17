module Parallel.Mandelbrot (mandelbrot) where

import Control.Parallel.Strategies

type Complex = (Double, Double)

mandelbrot :: Int -> Int -> Int -> [[Int]]
mandelbrot width height maxIter =
    parMap rdeepseq computeRow [0..height-1]
  where
    computeRow y =
        [ mandel (scaleX x, scaleY y) maxIter | x <- [0..width-1] ]

    scaleX x = (fromIntegral x / fromIntegral width) * 3.5 - 2.5
    scaleY y = (fromIntegral y / fromIntegral height) * 2.0 - 1.0

mandel :: Complex -> Int -> Int
mandel (cx, cy) maxIter = go 0 0 0
  where
    go x y iter
        | x*x + y*y > 4 = iter
        | iter >= maxIter = iter
        | otherwise =
            let x' = x*x - y*y + cx
                y' = 2*x*y + cy
            in go x' y' (iter + 1)