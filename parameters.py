#If it detects too little (misses mangroves)
MIN_SCORE = 0.2        # lower the threshold
TILE_SIZE = 400        # smaller window = more detail per patch
OVERLAP   = 0.4        # more overlap = fewer missed crowns at edges

#If it detects too much (noise, water, ground)
MIN_SCORE = 0.6        # stricter threshold
CROP_PADDING = 10      # tighter crops

#If you run out of RAM
TILE_SIZE = 300        # reduce window size