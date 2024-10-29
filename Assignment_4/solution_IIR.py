"""  Solution Assignment IIR
Running this script required the installation of:
    - pydicom
    - scikit-image
"""

# author:	Thomas Haslwanter
# date:		Aug-2024

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
    in_file = 'data/pleth.pickle'
    with open(in_file, 'rb') as fh:
        data = pickle.load(fh)
        rate = data['rate']
        pleth = data['pleth']

    time = np.arange(len(pleth))/rate
    nyq = rate/2        # Nyquist frequency

    order = 4

    # Lowpass-filter:
    upper_freq = 1.2
    (b, a) = signal.butter(order, upper_freq/nyq, btype='lowpass')
    lp_filtered = signal.lfilter(b, a, pleth, axis=0)

    # Highpass-filter:
    lower_freq = 1.5
    (b, a) = signal.butter(order, lower_freq/nyq, btype='highpass')
    hp_filtered = signal.lfilter(b, a, pleth, axis=0)

    # Bandpass filter:
    freq_band = np.r_[1.4, 5]   # lower and upper cutoff frequency (Hz)
    (b, a) = signal.butter(order, freq_band/nyq, btype='bandpass')
    bp_filtered = signal.lfilter(b, a, pleth, axis=0)   # Filter the data

    print(f'Low-pass: {upper_freq} Hz, High-pass: {lower_freq} Hz, Band-pass: {freq_band} Hz')

    # Plot the data
    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].plot(time, pleth, lw=0.8)
    axs[0].set_ylabel('original')
    axs[0].set_title('Plethysmographie')

    axs[1].plot(time, lp_filtered, lw=0.8)
    axs[1].set_ylabel('Low-pass\n"$\\it{Breathing}$"')
            #In this label the '\n' indicates a 'newline', and the text between
            # the '$' is LaTeX, used here to produce italic fonts

    axs[2].plot(time, hp_filtered, lw=0.8)
    axs[2].set_ylabel('High-pass\n"$\\it{Heartbeat}$"')
    axs[2].set_xlabel('Time (sec)')

    out_file = 'pleth.png'
    plt.savefig(out_file, dpi=300)
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

    # Get the data
    in_file = 'data/MR-MONO2-16-knee'
    ds = pydicom.dcmread(in_file)
    img = ds.pixel_array

    # Find edges
    edges = filters.sobel(img)
    threshold = np.max(edges)/3.5

    # Shot the data
    fig, axs = plt.subplots(1, 3)

    axs[0].imshow(img, cmap=plt.cm.bone)
    axs[0].set_title('Original data')

    axs[1].imshow(edges)
    axs[1].set_title('Sobel Edge Detection')

    axs[2].imshow(edges > threshold)
    axs[2].set_title('Strong Sobel Edges')
    out_file = 'knee_processed.jpg'
    plt.savefig(out_file)
    print(f'Processed knee saved to {out_file}')

    plt.show()


if __name__ == '__main__':
    # iir_filter()    # Ex 1
    # smoothing()     # Ex 2
    # pleth()         # Ex 3
    dicom_edges()   # Ex 4
