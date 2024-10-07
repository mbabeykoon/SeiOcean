
import numpy as np

# Constants
traces_per_shot = 5105
receivers_per_line = 638
num_lines = 8

########################################### Functions get_shot_data ########################################
def get_shot_data(shot_no, data):
    """
    Retrieve data for a specific shot, excluding the first receiver.
    Args:
        shot_no (int): The shot number (1-indexed).
        data (numpy.ndarray): The entire dataset.
    Returns:
        numpy.ndarray: Data for the specified shot.
    """
    start_idx = (shot_no - 1) * traces_per_shot + 1  # Skip the first receiver
    end_idx = shot_no * traces_per_shot
    return data[start_idx:end_idx]

########################################### Functions get_line_data ########################################
def get_line_data(shot_no, line_no, data):
    """
    Retrieve data for a specific line within a shot, excluding the first receiver.
    Args:
        shot_no (int): The shot number (1-indexed).
        line_no (int): The line number (1-indexed).
        data (numpy.ndarray): The entire dataset.
    Returns:
        numpy.ndarray: Data for the specified line within the shot.
    """
    shot_data = get_shot_data(shot_no, data)
    start_idx = (line_no - 1) * receivers_per_line
    end_idx = start_idx + receivers_per_line
    return shot_data[start_idx:end_idx]


########################################### Functions get_receivers_data ########################################

def get_receivers_data(shot_no, line_no, receiver_no, data):
    """
    Retrieve data for specific receivers within a line and shot, excluding the first receiver.
    Args:
        shot_no (int): The shot number (1-indexed).
        line_no (int): The line number (1-indexed).
        receiver_no (int or numpy.ndarray): The receiver number(s) (1-indexed).
        data (numpy.ndarray): The entire dataset.
    Returns:
        numpy.ndarray: Data for the specified receiver(s).
    """
    line_data = get_line_data(shot_no, line_no, data)
    
    if isinstance(receiver_no, (int, np.integer)):
        idx = receiver_no - 1
        return line_data[idx]
    elif isinstance(receiver_no, np.ndarray):
        idx = receiver_no - 1
        return line_data[idx]
    else:
        raise ValueError("receiver_no must be an integer or numpy array")
    

########################################### Functions select_line_data_portion ########################################

def select_line_data_portion(line_data, delta_t, receiver_range=None, time_range=None):
    """
    Select a portion of line data based on specified receiver and time ranges.
    
    Args:
        line_data (numpy.ndarray): The line data array (receivers x time samples).
        delta_t (float): Time step between samples in milliseconds.
        receiver_range (tuple): Optional. (start_receiver, end_receiver) to select a range of receivers.
        time_range (tuple): Optional. (start_time, end_time) in seconds to select a time range.
        
    Returns:
        tuple: (selected_signal, t, R)
            selected_signal (numpy.ndarray): The selected portion of the signal.
            t (numpy.ndarray): Time array corresponding to the selected signal.
            R (numpy.ndarray): Array of receiver numbers for the selected signal.
    """
    delta_t = 2
    signal_count = line_data.shape[1]  # Number of time stamps
    t = np.arange(0, signal_count) * delta_t * 1e-3  # Time in seconds
    
    receivers_per_line = line_data.shape[0]
    
    # Handle receiver range selection
    if receiver_range is None:
        receiver_range = (1, receivers_per_line)
    start_receiver, end_receiver = receiver_range
    start_receiver = max(1, start_receiver)
    end_receiver = min(receivers_per_line, end_receiver)
    
    R = np.arange(start_receiver, end_receiver + 1)
    selected_signal = line_data[start_receiver-1:end_receiver, :]

    # Handle time range selection
    if time_range is not None:
        start_time, end_time = time_range
        start_index = max(0, int(start_time / (delta_t * 1e-3)))
        end_index = min(signal_count, int(end_time / (delta_t * 1e-3)))
        t = t[start_index:end_index]
        selected_signal = selected_signal[:, start_index:end_index]

    return selected_signal,t,R


########################################### Function Matched Filter ########################################

def match_filter(signals, source):
    """
    Apply a matched filter to the input data using the given source.
    
    Returns:
    np.ndarray: Filtered data with the same shape as the input data
    """
    v = source[::-1]  # Reverse the source array
    N, M = signals.shape  # Get the dimensions of the data
    idx = np.argmax(v)  # Find the index of the maximum value in v
    
    # Initialize output array (has same shape as input data array)
    filtered_data = np.zeros_like(signals) 
    
    # Go through each receiver's signal and apply the filter 
    for i in range(M):
        u = signals[:, i]  # Signal from ith receiver
        
        # Convolve signal (u) with source (v)
        y = np.convolve(u, v, mode='full')  
        
        # Adjust the size and position of y
        y = y[idx:idx + N]
        
        # Re-scale the filtered signal to have the same amplitude as the signal 
        M0 = np.max(np.abs(u))  # Maximum amplitude of the original signal
        M1 = np.max(np.abs(y))  # Maximum amplitude of the filtered signal
        
        # Only adjust y if M1 is not zero
        if M1 != 0:
            y = y / M1 * M0
        else:
            # If M1 is zero, fill y with zeros or handle appropriately
            y = np.zeros_like(y)
        
        filtered_data[:, i] = y
    
    return filtered_data