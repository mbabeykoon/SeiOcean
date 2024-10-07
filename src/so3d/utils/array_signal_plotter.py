import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from so3d.utils.color_utils import redblue, redblue_cmap
from so3d.utils.data_utils import get_shot_data, get_line_data, get_receivers_data


########################################################################################################################


def plot_receiver_signals(receiver_data, receiver_arr, delta_t, x_lim=[0, 6], y_lim=[-1.1, 1.1], show_legend=True, shot_no=None, line_no=None):
    """
    Plot the normalized signal from a single receiver or an array of receivers.
    
    Parameters:
    - receiver_data: Data for the specified receiver(s) (1D or 2D array)
    - receiver_arr: Receiver number(s) (int or array-like)
    - delta_t: Time increment between timestamps
    - x_lim: X-axis limits for the plot
    - y_lim: Y-axis limits for the plot
    - show_legend: Boolean indicating whether to display the legend
    - shot_no: Shot number (optional, for title)
    - line_no: Line number (optional, for title)
    """
    # Ensure receiver_data is always 2D
    receiver_data = np.atleast_2d(receiver_data)
    
    # Ensure receiver_arr is always a list or array
    if isinstance(receiver_arr, (int, np.integer)):
        receiver_arr = [receiver_arr]
    
    signal_count = receiver_data.shape[1]  # Number of time stamps
    time = np.arange(0, signal_count) * delta_t * 1e-3  # Time in seconds
    
    print(f'Signal detected by receiver(s): {receiver_arr}')
    plt.figure(figsize=(12, 6))
    
    for i, r in enumerate(receiver_arr):
        normalized_signal = receiver_data[i] / np.max(np.abs(receiver_data[i]))
        plt.plot(time, normalized_signal, label=f'Receiver {r}')

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.ylabel('Amplitude (normalized)')
    plt.xlabel('Time (s)')

    if show_legend:
        plt.legend()

    if shot_no is not None and line_no is not None:
        plt.title(f'Normalized Signal from Receiver(s) (Shot {shot_no}, Line {line_no})')
    else:
        plt.title('Normalized Signal from Receiver(s)')

    plt.grid()
    plt.show()

########################################################################################################################

# Frequency plot of the signal (FFT)
def freq_plot(signal,source_signal, delta_t=2e-3, scaled_ss=False):
    """
    Plots the power spectral density (PSD) of the input signal.
    
    Parameters:
    - signal: 1D numpy array, the time-domain signal to be analyzed.
    - delta_t: float, time interval between samples (in seconds), default is 2 milliseconds.
    - scaled_ss: boolean, if True, the scaled power spectral density of the first signal is plotted.
    """
    
    # Assign the input signal to a variable `ss`
    ss = signal   
    
    # Get the length of the signal
    n = len(ss)     
    
    # Create a time array based on the number of samples and the delta_t
    time = np.arange(n) * delta_t

    # Perform Fast Fourier Transform (FFT) on the signal
    f_ss = np.fft.fft(ss)             
    
    # Compute the Power Spectral Density (PSD) from the FFT results
    psd = np.sqrt(f_ss * np.conj(f_ss) / n)   
   
    # Normalize PSD by dividing by its maximum value
    max_psd = max(psd) 
    psd = psd / max_psd

    # Compute corresponding frequencies
    freq = np.fft.fftfreq(n) / delta_t 
        
    # Extract positive frequencies
    pos_freq = freq[freq >= 0]
    
    # Compute PSD for the first signal (source_signal) for comparative analysis
    f_ss0 = np.fft.fft(source_signal)
    psd0 = np.sqrt(f_ss0 * np.conj(f_ss0) / n)
    max_psd0 = max(psd0)
    
    # Normalize the PSD of the first signal
    psd0 = psd0 / max_psd0
    
    # Plot the scaled PSD of the first signal if requested
    if scaled_ss:
        plt.plot(pos_freq, psd0[freq >= 0], label='s0')

    # Print the frequency corresponding to the maximum value of the PSD
    print(f'max_freq = {pos_freq[np.argmax(psd[freq >= 0])]:0.3f} Hz')
    
    # Plot the normalized PSD of the input signal
    plt.plot(pos_freq, psd[freq >= 0], label=f's/s0 = {np.abs(max_psd / max_psd0):.8f}')
    
    # Set the legend for the plot
    plt.legend()
    
    # Label the x-axis
    plt.xlabel('Frequency (Hz)')
    
    # Label the y-axis
    plt.ylabel('Power Spectral Density (PSD)')

########################################################################################################################

# Function to plot frequency for a given time interval
def freq_plot_tcut(signal,source_signal, t_min, t_max):
    """
    ​<light>Plots the frequency spectrum of a segment of the input signal defined by a time range.</light>​
    
    Parameters:
    - signal: 1D numpy array representing the time-domain signal.
    - t_min: float, minimum time (in seconds) for the segment to be analyzed.
    - t_max: float, maximum time (in seconds) for the segment to be analyzed.
    """
    
    # Assign the input signal to a variable `ss`
    ss = signal 
    
    # Get the length of the signal
    n = len(ss) 
    
    # Set the time interval (delta_t) between samples (fixed at 2 milliseconds)
    delta_t = 2e-3    
    
    # Create a time array based on the number of samples and the delta_t
    time = np.arange(n) * delta_t

    # Create a mask for the time array to isolate the desired time segment
    time_mask = (time < t_max) * (time > t_min)

    # Apply the time mask to the signal to isolate the segment of interest
    masked_sig = ss * time_mask
    
    # (Optional) Uncomment the next line to remove zero entries if desired
    # masked_sig = masked_sig[masked_sig != 0]

    # Call the frequency plot function on the masked signal
    freq_plot(masked_sig,source_signal)


########################################################################################################################



def plot_seismic(data, receiver_range=None, time_range=None, delta_t=2, fig_width=10, fig_height=10):
    """
    Plot seismic data for a given shot and line, with options to select specific receivers and time range.

    Parameters:
    - data: The seismic data to plot (2D numpy array)
    - shot_no: Shot number
    - line_no: Line number
    - receiver_range: Tuple of (start, end) receiver numbers to plot. If None, plots all receivers.
    - time_range: Tuple of (start, end) times in seconds to plot. If None, plots all time points.
    - delta_t: Time step in milliseconds
    - fig_width: Width of the figure in inches (default: 10)
    - fig_height: Height of the figure in inches (default: 10)
    """
    if data is None or len(data) == 0:
        print("No data provided. Unable to plot.")
        return

    signal_count = data.shape[1]  # Number of time stamps
    t = np.arange(0, signal_count) * delta_t * 1e-3  # Time in seconds
    
    receivers_per_line = data.shape[0]
    
    # Handle receiver range selection
    if receiver_range is None:
        receiver_range = (1, receivers_per_line)
    start_receiver, end_receiver = receiver_range
    start_receiver = max(1, start_receiver)
    end_receiver = min(receivers_per_line, end_receiver)
    
    R = np.arange(start_receiver, end_receiver + 1)
    selected_signal = data[start_receiver-1:end_receiver, :]

    # Handle time range selection
    if time_range is not None:
        start_time, end_time = time_range
        start_index = max(0, int(start_time / (delta_t * 1e-3)))
        end_index = min(signal_count, int(end_time / (delta_t * 1e-3)))
        t = t[start_index:end_index]
        selected_signal = selected_signal[:, start_index:end_index]

    # Create figure with specified or default size
    plt.figure(figsize=(fig_width, fig_height))
    plt.pcolormesh(R, t, selected_signal.T, shading='auto', cmap='seismic')
    plt.clim([-0.25, 0.25])  # set the color (amplitude) limit
    plt.ylim(t[-1], t[0])  # Reverse y-axis to have time increasing downwards
    plt.xlabel('Receiver No.')
    plt.ylabel('Time (s)')
    # plt.title(f'Signal for Shot {shot_no}, Line {line_no}, Receivers {start_receiver}-{end_receiver}')
    plt.colorbar(label='Amplitude')
    plt.show()

########################################################################################################################
def plot_shot_source_signal(shot_no, data, delta_t, x_lim=[0, 6], y_lim=[-1.1, 1.1]):
    """
    Plot the source signal for a specific shot.
    
    Args:
        shot_no (int): The shot number (1-indexed).
        data (numpy.ndarray): The entire dataset.
        delta_t (float): Time step between samples in microseconds.
        x_lim (list): X-axis limits for the plot in milliseconds.
        y_lim (list): Y-axis limits for the plot.
    """
    traces_per_shot = 5105
    # Calculate the index of the source signal for the given shot
    source_idx = (shot_no - 1) * traces_per_shot
    
    # Extract the source signal
    source_signal = data[source_idx]
    
    # Calculate time array in milliseconds
    signal_count = data.shape[1]  # number of time samples
    time = np.arange(0, signal_count) * delta_t * 1e-3  # convert to milliseconds
    
    # Normalize the signal
    normalized_signal = source_signal / np.max(np.abs(source_signal))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, normalized_signal)
    
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.ylabel('Amplitude (normalized)')
    plt.xlabel('Time (ms)')
    plt.title(f'Source Signal for Shot {shot_no}')
    
    print(f'Plotting source signal for shot {shot_no}')
    
    plt.grid(True)
    plt.show()

    ########################################################################################################################
    
def plot_signal_mf_comparison(s1, mf_1, mf_2, R, t):
    """
    Plot the original and matched filtered signals in a side-by-side format.
    
    Args:
        s1 (numpy.ndarray): The original signal data.
        mf_1 (numpy.ndarray): The first matched filtered signal data.
        mf_2 (numpy.ndarray): The second matched filtered signal data.
        R (numpy.ndarray): Array of receiver numbers.
        t (numpy.ndarray): Array of time values.
    """
    
    # Create a figure with 3 subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    # Create sample data ranges
    x1 = R  # Receiver numbers for s1
    y1 = t  # Time values for s1

    # Original signal
    im1 = ax1.pcolormesh(x1, y1, s1, shading='auto', cmap=redblue_cmap)
    ax1.set_ylim(max(y1), min(y1))  # Invert y-axis
    ax1.set_xlabel('Receiver Number')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Original')

    # Match filtered signal 1
    im2 = ax2.pcolormesh(x1, y1, mf_1, shading='auto', cmap=redblue_cmap)
    ax2.set_ylim(max(y1), min(y1))  # Invert y-axis
    ax2.set_xlabel('Receiver Number')
    ax2.set_title('MF_1')

    # Match filtered signal 2
    im3 = ax3.pcolormesh(x1, y1, mf_2, shading='auto', cmap=redblue_cmap)
    ax3.set_ylim(max(y1), min(y1))  # Invert y-axis
    ax3.set_xlabel('Receiver Number')
    ax3.set_title('MF_2')

    # Set uniform color limits for all plots
    vmin, vmax = -0.25, 0.25
    im1.set_clim([vmin, vmax])
    im2.set_clim([vmin, vmax])
    im3.set_clim([vmin, vmax])

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Amplitude')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()