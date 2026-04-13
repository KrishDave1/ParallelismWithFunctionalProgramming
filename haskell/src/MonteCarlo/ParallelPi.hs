{-
================================================================================
  MonteCarlo.ParallelPi — Parallel Monte Carlo Pi Estimation
================================================================================

PURPOSE:
  Parallel π estimation using two distinct Haskell parallelism mechanisms:
    1. ASYNC library   — Futures/Promises pattern
    2. STM             — Software Transactional Memory for result aggregation
  
  This problem is "embarrassingly parallel" — each sample is independent.
  The interesting FP aspects are HOW we distribute work and collect results.

KEY FP CONCEPTS DEMONSTRATED:

  1. FUTURES / PROMISES (Control.Concurrent.Async):
     'async' spawns a computation on a new thread and returns a handle.
     'wait' blocks until the computation is done and returns its result.
     
     This is the FUTURES/PROMISES pattern from concurrent programming,
     but made safe by Haskell's type system:
       - 'Async a' is a handle to a future result of type 'a'
       - 'wait :: Async a -> IO a' blocks until the result is available
       - Exceptions in the child thread are re-thrown to the parent
     
     COMPARISON WITH C++:
       C++:    auto future = std::async(std::launch::async, computeFunc);
               double result = future.get();
       
       Haskell: handle <- async (return $ computeFunc args)
                result <- wait handle
     
     The Haskell version is higher-level and handles exception propagation
     automatically.

  2. STM (Software Transactional Memory):
     STM provides COMPOSABLE atomic operations on shared state.
     Instead of locks (which don't compose), STM uses:
       - TVar: a transactional mutable variable
       - atomically: execute a transaction (reads/writes) atomically
       - retry: if a condition isn't met, automatically retry later
       - orElse: try one transaction, if it retries, try another
     
     KEY ADVANTAGES OVER LOCKS:
       a. COMPOSABILITY: You can combine two STM transactions into one,
          both executing atomically. With locks, composing two locked
          operations often causes deadlocks.
       b. NO DEADLOCKS: STM cannot deadlock (it uses optimistic concurrency).
       c. NO FORGOTTEN UNLOCKS: Transactions always complete, even on exceptions.
     
     COMPARISON WITH C++ MUTEX:
       C++:    std::mutex m;
               { std::lock_guard lock(m);
                 shared_counter += result; }  // Manual lock management
       
       Haskell: atomically $ modifyTVar' counter (+ result)
                -- Automatic, composable, deadlock-free

  3. SPLITTABLE PRNG FOR PARALLELISM:
     The key insight: we use 'split' to create independent PRNGs for each
     thread, from a single initial seed. This means:
       - No shared PRNG state between threads
       - No locks/synchronization for random number generation
       - Deterministic results (same seed → same split → same results)
     
     This is IMPOSSIBLE in imperative languages with mutable PRNGs.
     In C++, you must either:
       a. Lock the shared PRNG (slow — serializes random number generation)
       b. Create separate PRNGs per thread (possible but ad-hoc)
     
     Haskell's pure, splittable PRNGs make this natural and safe.

  4. THE ASYNC PATTERN: mapConcurrently
     'mapConcurrently f xs' applies 'f' to each element of 'xs' in parallel,
     using one OS thread per element. This is a higher-order function for
     parallelism — it abstracts over the threading boilerplate.

ARCHITECTURE:

  Version 1 (Async):
    ┌─────────────────────────────────────────┐
    │ Main Thread                              │
    │   split seed into N sub-seeds            │
    │   ┌──────┐ ┌──────┐ ┌──────┐            │
    │   │ Task │ │ Task │ │ Task │  async      │
    │   │ seed1│ │ seed2│ │ seedN│  spawn      │
    │   └──┬───┘ └──┬───┘ └──┬───┘            │
    │      │        │        │     wait all     │
    │   ┌──┴────────┴────────┴──┐              │
    │   │ Combine hit counts    │              │
    │   │ π = 4 × total / N    │              │
    │   └───────────────────────┘              │
    └─────────────────────────────────────────┘

  Version 2 (STM):
    ┌─────────────────────────────────────────┐
    │ TVar hitCounter = 0                      │
    │   ┌──────┐ ┌──────┐ ┌──────┐            │
    │   │ Task │ │ Task │ │ Task │  forked     │
    │   │  1   │ │  2   │ │  N   │  threads    │
    │   └──┬───┘ └──┬───┘ └──┬───┘            │
    │      │        │        │                 │
    │   atomically: modifyTVar hitCounter      │
    │                                          │
    │   Main waits for all tasks, reads TVar   │
    └─────────────────────────────────────────┘

================================================================================
-}

module MonteCarlo.ParallelPi
    ( parallelPiAsync
    , parallelPiSTM
    ) where

import MonteCarlo.SequentialPi (countHits)
import System.Random.SplitMix (SMGen, mkSMGen, splitSMGen)
import Control.Concurrent.Async (mapConcurrently)
import Control.Concurrent.STM
    ( TVar, newTVarIO, readTVarIO, atomically, modifyTVar' )
import Control.Concurrent (forkIO)
import Control.Monad (forM_)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Data.Word (Word64)

-- ============================================================================
-- Version 1: Async (Futures/Promises Pattern)
-- ============================================================================

{-|
  Estimate π using async (futures/promises).

  HOW IT WORKS:
    1. Create N independent PRNGs by repeatedly splitting the initial generator
    2. Use 'mapConcurrently' to run countHits on each PRNG in parallel
       Each call processes (totalSamples / numWorkers) samples
    3. Sum all hit counts and compute π

  'mapConcurrently' is a HIGHER-ORDER FUNCTION for parallelism:
    mapConcurrently :: (a -> IO b) -> [a] -> IO [b]
  It's the parallel version of 'mapM' — applies a function to each element
  concurrently, collecting all results.

  PARAMETERS:
    - numWorkers:   number of parallel threads (ideally = numCores)
    - totalSamples: total number of Monte Carlo samples
    - seed:         initial PRNG seed

  DETERMINISM:
    Because we split the PRNG deterministically and each worker processes
    a fixed number of samples, the result is the SAME for a given
    (seed, numWorkers, totalSamples) triple. Deterministic parallelism!
-}
parallelPiAsync :: Int -> Int -> Word64 -> IO Double
parallelPiAsync numWorkers totalSamples seed = do
    let samplesPerWorker = totalSamples `div` numWorkers
        -- Create N independent generators by splitting
        gens = makeGenerators numWorkers (mkSMGen seed)
        -- Each worker gets (samplesPerWorker, its own generator)
        tasks = zip (repeat samplesPerWorker) gens
    
    -- mapConcurrently: parallel map over IO actions
    -- Each element of 'tasks' is processed in a separate thread
    results <- mapConcurrently
        (\(n, gen) -> return $! countHits n gen)  -- $! forces strict evaluation
        tasks
    
    -- Aggregate results: sum all hit counts
    let totalHits = sum results
    return $ 4.0 * fromIntegral totalHits / fromIntegral (numWorkers * samplesPerWorker)

-- ============================================================================
-- Version 2: STM (Software Transactional Memory)
-- ============================================================================

{-|
  Estimate π using STM for shared result aggregation.

  HOW IT WORKS:
    1. Create a TVar (transactional variable) initialized to 0
    2. Fork N worker threads, each with its own PRNG
    3. Each worker counts hits and atomically adds to the TVar
    4. Main thread waits for all workers, then reads the TVar

  WHY STM HERE?
    This demonstrates an ALTERNATIVE coordination strategy:
    - Async: results flow back to parent via return values
    - STM:   results accumulate in shared transactional memory
    
    STM is more appropriate when:
    - Workers produce partial results that should be aggregated incrementally
    - You need to combine results from workers as they complete
    - The aggregation itself requires complex atomic operations

  COMPARISON WITH ASYNC VERSION:
    - Functionally equivalent for this problem
    - STM adds overhead (transaction bookkeeping) but enables richer patterns
    - STM shines when aggregation is complex (e.g., merging data structures)

  COMPARISON WITH C++ MUTEXES:
    C++: Each worker locks a mutex, updates a counter, unlocks.
         If you forget to unlock → deadlock. If two mutexes → potential deadlock.
    STM: atomically { modifyTVar counter (+hits) } — always correct, composable.
-}
parallelPiSTM :: Int -> Int -> Word64 -> IO Double
parallelPiSTM numWorkers totalSamples seed = do
    -- Create a shared TVar for accumulating hits
    -- TVar = Transactional Variable (mutable, but only modifiable inside 'atomically')
    hitCounter <- newTVarIO (0 :: Int)
    
    -- Create synchronization barriers (one MVar per worker)
    -- Workers signal completion by putting () into their MVar
    barriers <- mapM (\_ -> newEmptyMVar) [1..numWorkers]
    
    let samplesPerWorker = totalSamples `div` numWorkers
        gens = makeGenerators numWorkers (mkSMGen seed)
    
    -- Fork worker threads
    forM_ (zip gens barriers) $ \(gen, barrier) -> forkIO $ do
        -- Each worker independently counts hits (no shared state here!)
        let hits = countHits samplesPerWorker gen
        -- Atomically add hits to the shared counter
        -- 'atomically' ensures this is an atomic read-modify-write
        -- No locks, no deadlocks, no forgotten unlocks
        atomically $ modifyTVar' hitCounter (+ hits)
        -- Signal completion
        putMVar barrier ()
    
    -- Wait for all workers to finish
    mapM_ takeMVar barriers
    
    -- Read the final count
    totalHits <- readTVarIO hitCounter
    return $ 4.0 * fromIntegral totalHits / fromIntegral (numWorkers * samplesPerWorker)

-- ============================================================================
-- Helper: Create N independent generators by splitting
-- ============================================================================

{-|
  Create N independent PRNGs from a single generator using 'split'.

  PURE RNG SPLITTING:
    gen0 → split → (gen1a, gen1b)
    gen1b → split → (gen2a, gen2b)
    gen2b → split → (gen3a, gen3b)
    ...
    We take the 'a' generators and give one to each worker.
    All generators are statistically independent.

  This is a uniquely FP technique — imperative languages don't have
  splittable PRNGs by default.
-}
makeGenerators :: Int -> SMGen -> [SMGen]
makeGenerators 0 _   = []
makeGenerators 1 gen = [gen]
makeGenerators n gen =
    let (gen1, gen2) = splitSMGen gen
    in gen1 : makeGenerators (n - 1) gen2
