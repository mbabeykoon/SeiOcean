# segy_reader.py

import numpy as np
import segyio

def read_segy(sgyfile, textheader=None):
    dataout = None
    sampint = None
    textheader_output = None
    
    with segyio.open(sgyfile, "r", ignore_geometry=True) as f:
        # Get the number of traces and sample size
        num_traces = f.tracecount      # Number of traces 
        num_samples = f.samples.size    # Number of samples per trace
        print(f'No of traces = {num_traces}, No of samples per trace = {num_samples}')
        
        # Initialize dataout array
        dataout = np.zeros((num_traces, num_samples))
        
        # Populate the dataout with traces
        for i in range(num_traces):
            dataout[i, :] = f.trace[i]  # Fill each row with the trace data
        
        # Extracting sample interval
        delta_t = f.samples[1] - f.samples[0]  # Assuming regular sampling
        
        # Extracting text header if requested
        if textheader == 'yes':
            textheader_output = segyio.tools.wrap(f.text[0])
    return dataout, delta_t, textheader_output