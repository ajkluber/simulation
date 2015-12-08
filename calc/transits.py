import numpy as np


def partition_dtraj(dtraj, state1, state2): 
    """ Partition the trajectory into transits bewteen states and intra-state dwells.

    Parameters
    ----------
    dtraj : np.ndarray, or list
        Discrete trajectory; timeseries of state-occupancies. State elements .
    
    state1 : int
        state to  

    state2 : int



    # discard initial frames that aren't in either state1 or state2.

    Returns
    -------
    dwells : list 
    # collect the first frame index and length of:
    #   - dwells in state1
    #   - dwells in state2
    #   - transits from state1 to state2
    #   - transits from state2 to state1


    """
    nframes =  len(dtraj)
    state_idx = {state1:0, state2:1}
    states = [state1, state2]
    
    # discard frames that are in transit region.
    discard_frames = 0
    for i in range(nframes):
        if dtraj[i] in states:
            prev_state = dtraj[i]
            from_state = dtraj[i]
            break
        else:
            discard_frames += 1

    # collect list of [start_idx, length] for dwells and transits
    dwells = [[], []]
    transits = [[],[]]

    dwell_len = 1
    dwell_start_idx = discard_frames
    for i in range(discard_frames, nframes):
        curr_state = dtraj[i]
        if curr_state not in states:
            # current frame is in transit region
            if prev_state not in states:
                # continuing transit
                transit_len += 1
            else:
                # starting potential transit
                transit_len = 1
                transit_start_idx = i
        else:
            # current frame is in state1 or state2 
            if prev_state == curr_state:
                # extend dwell
                dwell_len += 1
            elif (prev_state != curr_state) and (prev_state not in states):
                if from_state == curr_state:
                    # failed transit; extend dwell time
                    dwell_len += transit_len + 1
                else:
                    # successful transit! starting new dwell
                    transits[state_idx[from_state]].append([transit_start_idx, transit_len])
                    dwells[state_idx[from_state]].append([dwell_start_idx, dwell_len])
                    dwell_start_idx = i
                    dwell_len = 1
                    from_state = curr_state
            else:
                # transit of length 0! this should never happen.
                raise IOError("This trajectory has transits of zero frames!")

        prev_state = curr_state
    dwells1 = np.array(dwells[0])
    dwells2 = np.array(dwells[1])
    transits12 = np.array(transits[0])
    transits21 = np.array(transits[1])
    return dwells1, dwells2, transits12, transits21

