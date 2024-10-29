""" Solution to Assignment Events, BSA """

# author:   Thomas Haslwanter
# date:     Nov-2024

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, io


def simulate_experiment() -> None:
    """ Part 1: Simulate experimental data """
    # Set the parameters
    noise_amp = 0.1
    data_range = 15
    step_size = 5
    num_data = 500
    out_file = './data/HorPos.txt'

    # Simulate the data
    part = np.zeros(num_data)
    offsets = np.arange(-data_range, data_range+1, step_size)
    offsets = np.hstack( (0, offsets, 0))   # Start and end at 0

    data = noise_amp * np.random.randn(len(part))
    for offset in offsets[1:]:
        data = np.hstack((data, offset + noise_amp * np.random.randn(len(part))))

    # Show the data
    plt.plot(data)
    plt.show()

    # Save to out_file
    np.savetxt(out_file, data)
    print(f'Simulated data saved to {out_file}.')


def emg_activity() -> None:
    """ Part 2 - EMG Analysis: Find the mean duration of the EMG-activity
    Sample-data from  https://www.shimmersensing.com/support/sample-data/
    """

    # ------------- To do ------------------------
    # Here you have to find the onsets and offsets
    # ....
    # ....
    # ....
    # --------------------------------------------

    # Eliminate filter artefacts at the beginning of the file
    beginning = 2000   # points
    onset = onset[onset > beginning]
    offset = offset[offset > beginning]

    # Make sure we start with an onset, and end with an offset
    if onset[0] > offset[0]:
        offset = offset[1:]

    if offset[-1] < onset[-1]:
        onset = onset[:-1]

    # "assert" runs through if everything is ok
    assert(len(onset) == len(offset))

    # Find the durations
    # Eliminate bursts of muscle activity that are too short
    # to correspond to a real movement
    min_interval = 0.5  # [sec]

    # Calculate contraction times
    dt = 1/rate
    interval = (offset-onset)*dt

    # Eliminate short drops
    valid = interval > min_interval
    interval = interval[valid]
    onset = onset[valid]
    offset = offset[valid]

    # Show the resulting on- and offsets
    plt.plot(time, smoothed, label='activity')
    plt.xlabel('Time [sec]')
    plt.ylabel('EMG-Envelope [mV]')

    plt.plot(onset/rate, np.zeros_like(onset), '*', label='start')
    plt.plot(offset/rate, np.zeros_like(offset), 'o', label='end')

    disp_txt = 'Contraction time = ' +\
              f'{np.mean(interval):.2f} ± {np.std(interval, ddof=1):#.2f} s'
    plt.text(50, 1.5, disp_txt)
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(time, emg, lw=0.5)
    axs[0].set_ylabel('EMG')

    axs[1].plot(time, smoothed)
    axs[1].set_xlabel('Time (sec)')
    axs[1].set_ylabel('Control signal')
    axs[1].set_xlim([10, 40])

    out_file = 'control_signal.jpg'
    plt.savefig(out_file)
    print(f'control signal image saved to {out_file}')
    plt.show()



def gait_analysis() -> None:
    """ Part 3: Gait analysis
    Find "Heelstrike" and "Toe-off" events from ground-reaction forces
    """

    # Load the new data
    in_file = 'Data/GroundReactionForce.mat'
    raw_data = io.loadmat(in_file, squeeze_me=True)
    grf = raw_data['GRF']

    # check where the Z-axis of the GRF is above the threshold
    threshold = 15
    contact = grf[:, 3] > threshold
    time = grf[:, 0]

    # heel strike occours when the contact signal changes from 0 to 1
    idx = {}
    idx['heel_strike'] = np.where( np.diff(contact*1.) == 1)[0]

    # toe off occours when the contact signal changes from 1 to 0
    idx['toe_off'] = np.where( np.diff(contact*1.) == -1)[0]

    # Plot the data
    plt.plot(time, grf[:,3], label='grf')
    plt.plot(time[idx['heel_strike']], grf[idx['heel_strike'], 3],
            'o', label='HS')
    plt.plot(time[idx['toe_off']], grf[idx['toe_off'], 3],
            '*', label='TO')

    # Format the plot
    plt.xlabel('Time [sec]')
    plt.ylabel('Force [N]')
    plt.title('Ground reaction force')
    plt.legend()
    plt.show()


def ecg_analysis() -> None:
    """ Part 4: R-peak detection 
    Find the heartbeats, and calculate the average bpm-rate
    """
    # ------------- To do ------------------------
    # Here you have to find the ECG-peaks, and the heart-rate
    # ....
    # ....
    # ....
    # --------------------------------------------


    # Plot the data
    plt.plot(time, ecg)
    plt.xlabel('Time [sec]')
    plt.ylabel('ECG')
    plt.title('R-Detection')
    plt.plot(time[peaks_index],ecg[peaks_index], 'o')

    disp_txt = 'Heartrate = ' +\
            f'{mean_heart_rate:.3f} ±  {std_heart_rate:.2f} min^{-1}'

    plt.text(time[int(len(time)/2)], -2.5, disp_txt)
    plt.show()


if __name__ == "__main__":
    simulate_experiment()   # Ex 1
    # emg_activity()          # Ex 2
    gait_analysis()         # Ex 3
    # ecg_analysis()          # Ex 4

