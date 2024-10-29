"""  Solution Assignment IIR
Running this script required the installation of:
    - pydicom
    - scikit-image
"""

# author:	Thomas Haslwanter
# date:		Nov-2023

# Import the standard packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import io        # for simple Matlab files
import pydicom              # for DICOM images
from skimage import filters
import pickle


def iir_filter() -> None:
    """ Exercise 1: IIR-filter """
    # Set the parameters
    a = [1, -0.5095]
    b = [0.7548, -0.7548]
    duration = 100
    t0 = 10

    # Generate 'impulse' and 'step'
    impulse = np.zeros(duration)
    impulse[t0] = 1

    step = np.zeros(duration)
    step[t0:] = 1

    # Filter the data
    impulse_filtered = signal.lfilter(b, a, impulse)
    step_filtered = signal.lfilter(b, a, step)

    # Plot signal and response
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].plot(impulse, '*', label='input')
    axs[0].plot(impulse_filtered, label='output')
    axs[0].set_ylabel('Impulse')
    axs[0].margins(x=0)
    axs[0].legend()

    axs[1].plot(step, '*', label='input')
    axs[1].plot(step_filtered, label='output')
    axs[1].set_xlabel('Points')
    axs[1].set_ylabel('Step')
    axs[1].legend()

    plt.show()


def smoothing() -> None:
    """ Smooth ECG-Date with FIR and IIR-filters """

    # Set the parameters
    in_file = r'data/ecg_hfn.mat'
    sample_rate = 1000
    iir_order = 2
    fir_order = 20  # This value has been determined experimentally
                    # values > 20 give a similar performance as a 2nd-order IIR
    cutoff_freq = 25

    # Get the data
    data = io.loadmat(in_file)
    ecg = data['ecg_hfn']

    # Find the filter parameters
    nyq = sample_rate/2
    cutoff_n = cutoff_freq/nyq
    time = np.arange(0, len(ecg))/sample_rate

    b_fir = signal.firwin(fir_order, cutoff_n, pass_zero='lowpass')
    (b_iir, a_iir) = signal.butter(iir_order, cutoff_n, btype='lowpass')

    # Filter the data
    ecg_fir = signal.lfilter(b_fir, 1, ecg, axis=0)
    ecg_iir = signal.lfilter(b_iir, a_iir, ecg, axis=0)

    print(f'The orders of the FIR/IIR-filter were {fir_order} and {iir_order}, respectively.')

    # Plot the results
    plt.plot(time, ecg, lw=0.5, label='ecg')
    plt.plot(time, ecg_fir, lw=0.5, label='fir')
    plt.plot(time, ecg_iir, lw=0.5, label='iir')
    plt.xlabel('Time (sec)')
    plt.ylabel('(arbitrary unit)')
    plt.legend()
    plt.xlim([0, 1])        # zoom in, to see the details
    plt.show()


def pleth() -> None:
    """ Extract breathing contributions in plethysmography data """

    # Get the data

    # From the original Matlab file:
    # import mat4py               # for more complicated Matlab files
    # in_file = 'data/monitor.mat'
    # data = mat4py.loadmat(in_file)
    # rate = data['monitor']['param']['samplingrate']
    # pleth = data['monitor']['signals']['pleth']

    # From a pickled dictionary:
    # "pickle" is a Python tool that lets you save any Python object.
    # Here it is used to retrieve a Python-'dictionary', containing
    # parameters and data values


    ## [xxx ----------------- To-do ---------------------------------
    # For the plots you have to generate the following variables:
    #   * time
    #   * pleth (original data)
    #   * lp_filtered (lowpass)
    #   * hp_filtered (highpass)
    #   * bp_filtered (bandpass)
    # ------------------------------------------------------------- xxx]

    # Plot the data
    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].plot(time, pleth, lw=0.8)
    axs[0].set_ylabel('original')
    axs[0].set_title('Plethysmographie')

    axs[1].plot(time, lp_filtered, lw=0.8)
    axs[1].set_ylabel('Low-pass\n"$\it{Breathing}$"')
            #In this label the '\n' indicates a 'newline', and the text between
            # the '$' is LaTeX, used here to produce italic fonts

    axs[2].plot(time, hp_filtered, lw=0.8)
    axs[2].set_ylabel('High-pass\n"$\it{Heartbeat}$"')
    axs[2].set_xlabel('Time (sec)')

    out_file = 'pleth.png'
    plt.savefig(out_file, dpi=200)
    # Always let the user know when an existing file has been modified, or when
    # a new file is generated.
    print(f'Plethysmography signals saved to {out_file}')
    plt.show()

    plt.plot(time, bp_filtered, lw=0.8)
    plt.xlabel('Time (sec)')
    plt.ylabel('Band-pass filtered Plethysmography')
    plt.show()


def dicom_edges() -> None:
    """ Read in DICOM-data, and find clear edges """
    ## [xxx ----------------- To-do ----------------------------- xxx]

if __name__ == '__main__':
    iir_filter()    # Ex 1
    smoothing()     # Ex 2
    pleth()         # Ex 3
    dicom_edges()   # Ex 4
