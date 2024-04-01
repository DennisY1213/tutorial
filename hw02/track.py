import numpy as np
# Constants
GRID_SIZE = 32

        
def build_track():
    ###Build the turn
    track = np.zeros([GRID_SIZE, GRID_SIZE])

    ##Start line
    track[-1, :] = 1
    ##Finish line
    track[:9, -1] = 2

    track[0,:17] = -1
    track[1,:14] = -1
    track[2,:13] = -1
    track[3,:12] = -1
    track[4:8,:12] = -1
    track[8,:13] = -1
    track[9,:14] = -1
    track[10:15,:15] = -1
    idx = 14
    for r in range(15, 30):
        track[r, :idx] = -1
        idx -= 1
    track[9, -2:] = -1
    track[10, -5:] = -1
    track[11, -6:] = -1
    track[12, -8:] = -1
    track[13:, -9:] = -1

    
    return track
        
    
# print(track)

