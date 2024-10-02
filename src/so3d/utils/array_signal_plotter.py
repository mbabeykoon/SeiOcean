import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_signal(receiver_no, data, delta_t, x_lim=[0, 6], y_lim=[-1.1, 1.1], show_legend=True):
    """
    Plot the normalized signal from a receiver array.

    Parameters:
    - receiver_no: List of receiver indices to plot.
    - data: 2D numpy array with signal data.
    - delta_t: Time increment between timestamps.
    - x_lim: X-axis limits for the plot.
    - y_lim: Y-axis limits for the plot.
    - show_legend: Boolean indicating whether to display the legend.
    """
    signal_count = data.shape[1]  # Number of time stamps
    time = np.arange(0, signal_count) * delta_t * 1e-3  # Time in ms
    print(f'Signal detected by receivers: {receiver_no}')
    
    for r in receiver_no:
        plt.plot(time, data[r] / max(abs(data[r])), label=f'Receiver {r}')
    
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.ylabel('Amplitude (normalized)')
    plt.xlabel('Time (s)')
    
    # Show legend based on the `show_legend` parameter
    if show_legend:
        plt.legend()
        
    plt.title('Normalized Signal from Receiver Array')
    plt.grid()
    plt.show()

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