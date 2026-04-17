module Utils.Timing (timeIt) where

import Data.Time.Clock

timeIt :: IO a -> IO a
timeIt action = do
    start <- getCurrentTime
    result <- action
    end <- getCurrentTime
    print (diffUTCTime end start)
    return result