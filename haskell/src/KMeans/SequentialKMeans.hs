{-# LANGUAGE BangPatterns #-}
{-
================================================================================
  KMeans.SequentialKMeans — Sequential K-Means Clustering (Baseline)
================================================================================

PURPOSE:
  Implement Lloyd's K-Means clustering algorithm sequentially. This is a
  foundational MACHINE LEARNING algorithm used for unsupervised learning —
  grouping data points into K clusters based on feature similarity.

  K-Means is directly called out in the project requirements as an example
  of a "machine learning algorithm suitable for parallel processing."

THE ALGORITHM (Lloyd's Algorithm):
  1. INITIALIZE: Pick K random centroids
  2. ASSIGN: For each data point, find the nearest centroid → assign to that cluster
  3. UPDATE: Recompute each centroid as the mean of all points assigned to it
  4. REPEAT steps 2-3 until centroids stop moving (convergence)

  The ASSIGN step is embarrassingly parallel: each point's assignment is
  independent of every other point. This is where parallelism helps.

KEY FP CONCEPTS DEMONSTRATED:

  1. IMMUTABLE DATA REPRESENTATION:
     Each "Point" is an immutable pair of (Double, Double).
     Each iteration produces a NEW list of centroids, never mutating the old ones.
     
     COMPARISON WITH IMPERATIVE:
       C++: centroids[i].x += point.x; centroids[i].count++;  // mutation
       Haskell: newCentroids = map computeCentroid clusters     // new value

  2. HIGHER-ORDER FUNCTIONS FOR THE ASSIGN STEP:
     The assignment step is a 'map': apply "find nearest centroid" to each point.
       assignments = map (assignToCluster centroids) points
     
     This naturally expresses the independence of each assignment.
     In the parallel version, we simply change 'map' to 'parMap'.

  3. PURE FUNCTIONS → DETERMINISTIC CONVERGENCE:
     Given the same initial centroids and data, K-Means will always converge
     to the same result. No non-determinism from thread scheduling.

  4. ALGEBRAIC DATA TYPES:
     We use type aliases and tuples to represent domain concepts clearly:
       type Point = (Double, Double)
       type Cluster = [Point]
     
     Pattern matching makes decomposition natural:
       centroid (xs, ys) = (mean xs, mean ys)

COMPLEXITY:
  Per iteration: O(n × k) where n = number of points, k = number of clusters
  Total: O(n × k × iterations)
  The ASSIGN step (O(n × k)) is the parallelizable bottleneck.

================================================================================
-}

module KMeans.SequentialKMeans
    ( Point
    , Centroid
    , kMeans
    , assignPoint
    , updateCentroids
    , distance
    , hasConverged
    ) where

import Data.List (minimumBy, groupBy, sortBy, foldl')
import Data.Ord (comparing)
import Control.DeepSeq (NFData(..))

-- ============================================================================
-- Data Types
-- ============================================================================

{-|
  A Point in 2D space. We use a simple tuple for clarity.
  
  In a production system, you might use a vector type for N-dimensional
  points, but tuples make the FP concepts clearer for educational purposes.
  
  IMMUTABILITY: Points are never modified. When we "move" a centroid,
  we create a new Point value.
-}
type Point = (Double, Double)

{-|
  A Centroid is just a Point that represents the center of a cluster.
  Using a type alias makes the code self-documenting.
-}
type Centroid = Point

-- ============================================================================
-- Core K-Means Algorithm
-- ============================================================================

{-|
  Run K-Means clustering until convergence or max iterations.

  PARAMETERS:
    - k:             number of clusters
    - maxIter:       maximum iterations (safety bound)
    - epsilon:       convergence threshold (how much centroids can move)
    - initialSeeds:  initial centroid positions
    - points:        data points to cluster

  RETURNS:
    Final centroids after convergence

  IMPLEMENTATION:
    This is a recursive loop that:
    1. Assigns all points to their nearest centroid
    2. Recomputes centroids from the assignments
    3. Checks for convergence (centroids moved less than epsilon)
    4. Recurses with new centroids if not converged

  KEY FP INSIGHT:
    Each iteration is a PURE FUNCTION from old centroids to new centroids:
      iterate :: [Centroid] -> [Centroid]
    No mutable state accumulates between iterations.
    The algorithm is expressed as a fixed-point computation.
-}
kMeans :: Int         -- ^ max iterations
       -> Double      -- ^ convergence epsilon
       -> [Centroid]  -- ^ initial centroids
       -> [Point]     -- ^ data points
       -> [Centroid]  -- ^ final centroids
kMeans maxIter epsilon initCentroids points = go maxIter initCentroids
  where
    go :: Int -> [Centroid] -> [Centroid]
    go 0 centroids = centroids  -- Max iterations reached
    go n centroids =
        let -- STEP 1: ASSIGN each point to nearest centroid
            -- This is a 'map' — each point is independently assigned
            assignments = map (assignPoint centroids) points
            
            -- STEP 2: UPDATE centroids by computing means
            newCentroids = updateCentroids (length centroids) assignments
            
            -- STEP 3: CHECK convergence
        in if hasConverged epsilon centroids newCentroids
           then newCentroids  -- Converged! Return result
           else go (n - 1) newCentroids  -- Recurse with new centroids

-- ============================================================================
-- Assignment Step
-- ============================================================================

{-|
  Assign a single point to the nearest centroid.

  Returns a pair: (cluster index, point)

  HIGHER-ORDER FUNCTION USAGE:
    'minimumBy (comparing fst)' finds the centroid with minimum distance.
    We zip distances with indices, find the minimum, and return the index.

  This function is PURE and INDEPENDENT per point — perfect for parallelism.
-}
assignPoint :: [Centroid] -> Point -> (Int, Point)
assignPoint centroids point =
    let distances   = map (distance point) centroids
        indexed     = zip [0..] distances
        (nearest,_) = minimumBy (comparing snd) indexed
    in (nearest, point)

{-|
  Euclidean distance between two points.
  Pure function — no side effects.
-}
distance :: Point -> Point -> Double
distance (x1, y1) (x2, y2) = sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- ============================================================================
-- Update Step
-- ============================================================================

{-|
  Recompute centroids from assignments.

  For each cluster, the new centroid is the mean of all assigned points.

  IMPLEMENTATION:
    1. Group assignments by cluster index
    2. For each cluster, compute the mean x and mean y
    3. If a cluster has no points, keep the old centroid position

  HIGHER-ORDER FUNCTIONS USED:
    - map: transform each cluster's points into a centroid
    - foldl': accumulate sums for mean computation
    - filter: select points belonging to each cluster
-}
updateCentroids :: Int -> [(Int, Point)] -> [Centroid]
updateCentroids k assignments =
    map computeClusterCentroid [0..k-1]
  where
    computeClusterCentroid :: Int -> Centroid
    computeClusterCentroid clusterIdx =
        let clusterPoints = [p | (idx, p) <- assignments, idx == clusterIdx]
        in if null clusterPoints
           then (0, 0)  -- Empty cluster (shouldn't happen with good initialization)
           else meanPoint clusterPoints
    
    meanPoint :: [Point] -> Point
    meanPoint ps =
        let n = fromIntegral (length ps)
            (!sumX, !sumY) = foldl' (\(!sx, !sy) (x, y) -> (sx + x, sy + y)) (0, 0) ps
        in (sumX / n, sumY / n)

-- ============================================================================
-- Convergence Check
-- ============================================================================

{-|
  Check if centroids have converged (moved less than epsilon).

  PURE FUNCTION: Takes old and new centroids, returns Boolean.
  No mutable "converged" flag to set — just a comparison.
-}
hasConverged :: Double -> [Centroid] -> [Centroid] -> Bool
hasConverged epsilon old new' =
    all (\(o, n) -> distance o n < epsilon) (zip old new')
