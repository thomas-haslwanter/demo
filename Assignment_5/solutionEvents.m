%Solution_Assignment_Events  
% ***************************

% autor: Thomas Haslwanter
% date:  Nov-2023


%% ---------- Exercise 1 ----------------------------------
% Simulate experimental data

% Set the parameters
noiseAmp = 0.1;
dataRange = 15;
stepSize = 5;
numData = 500;

% Steps
stepData = [];
for ii = -dataRange:stepSize:dataRange
    newData = ii*ones(1,numData)+ noiseAmp*randn(1,numData);
    stepData = [stepData newData];
end

% Beginning and end
HorPos = [noiseAmp*randn(1,numData), stepData noiseAmp*randn(1,numData)];

% Show the data
plot(HorPos);
xlabel('Points');
ylabel('Simulated Position');
shg

% Save the variable "HorPos" to a MAT-file with the same name
outFile = 'data\HorPos.mat';
save(outFile, 'HorPos');
disp(['Exercise 1: Simulated data written to ' outFile ', ready for "Exercise 2"']);
disp('Hit any key to continue');
pause
close('all');


%% ---------- Exercise 2: EMG Analysis ----------------------
% analyze_emg  Find the mean duration of the EMG-activity
% Sample-data from: https://www.shimmersensing.com/support/sample-data/

% Get the data
in_file = 'data\Shimmer3_EMG_Calibrated.csv'
% Skip 3 rows, and 3 columns
% The two columns show EMG from forearm and biceps
data = readmatrix(in_file, 'Range', [4, 4]);
data = data(:,1:2);     % We don't need the last column of 'NaN'
rate = 512;             % [Hz]

% Show them
emg = data(:,1);
time = (1:length(emg))/rate;
plot(time, emg)

axis('tight');
xlabel('Time [sec]');
ylabel('EMG [mV]');
title('Original Data');
shg
disp('Hit any key to continue');
pause

% High-pass filter the data, rectify and smooth them
nyq = rate/2;
[b,a] = butter(5, 1/nyq, 'high');   % cutoff at 1 Hz
filtered = filter(b,a,emg);
smoothed = savgol(abs(filtered), 3, 101);

plot(time, smoothed, 'r');

axis('tight');
xlabel('Time [sec]');
ylabel('EMG-Envelope [mV]')
title('Offset Removed, with Highpass-Filter')
shg
disp('Hit any key to continue');
pause

% Find onsets and offsets
threshold = 0.05;
activity = smoothed > threshold;
onset = find(diff(activity)==1);
offset = find(diff(activity)==-1);

% Eliminate filter artefacts at the beginning of the file
beginning = 2000;   % points
onset = onset(onset>beginning);
offset = offset(offset>beginning);

% Make sure we start with an onset, and end with an offset
if onset(1) > offset(1)
    offset = offset(2:end);
end

if offset(end) < onset(end)
    onset = onset(1:end-1);
end

% "assert" runs through if everything is ok, produces an error/stop otherwise
assert(length(onset) == length(offset));

% Find the durations
% Eliminate bursts of muscle activity that are too short
% to correspond to a real movement
min_interval = 0.5; % [sec]

% Calculate contraction times
dt = 1/rate;
interval = (offset-onset)*dt;

% Eliminate short drops
short_drops = interval < min_interval;
interval(short_drops) = [];
onset(short_drops) = [];
offset(short_drops) = [];

% Show the resulting on- and offsets
hold on
plot(onset/rate, zeros(size(onset)), 'b*');
plot(offset/rate, zeros(size(offset)), 'ro');
legend('Data', 'EMG-Activity: Start', 'EMG-Activity: End');
dispTxt = sprintf('Contraction time = %.2f ± %.2f s', mean(interval), ...
        std(interval));
text(50, 1.5, dispTxt);
 
shg
disp('Hit any key to continue, to Exercise 3');
pause
close('all');


%% ---------- Exercise 3: Gait events ----------------------
% Find "Heelstrike" and "Toe-off" events from ground-reaction forces

% Load the new data
inFile = 'data\GroundReactionForce.mat';
load(inFile);

% check where the Z-axis of the GRF is above the threshold
threshold = 15;
isContact = GRF(:,4) > threshold;
time = GRF(:,1);

% heel strike occours when the contact signal changes from 0 to 1
heelStrikeIndex = find( diff(isContact)==1 );

% toe off occours when the contact signal changes from 1 to 0
toeOffIndex = find( diff(isContact)==-1 );

% Plot the data
plot(time, GRF(:,4));
hold on
plot(time(heelStrikeIndex), GRF(heelStrikeIndex,4), 'og');
plot(time(toeOffIndex),     GRF(toeOffIndex,4),     'or');

% Format the plot
xlabel('Time [sec]')
ylabel('Force [N]')
title('Ground reaction force');
legend('grf','HS','TO')

shg
disp('Hit any key to continue, to Ex 4.4');
pause
close('all');


%% ---------- Exercise 4: R-peak detection ----------------
% Find the heartbeats, and calculate the average bpm-rate

% Set the parameters
inFile = 'Data\ecg_hfn.mat';
fs = 1000;  % [Hz]
threshold = 1.5; % min value signal has to be for peak detection

% Get the data
load(inFile);
time = 0:1/fs:(length(ecg_hfn)-1)/fs;

% Peaks can be found by checking the 1st derivative for zero-crossings
ecg_derivative = savgol(ecg_hfn,3,51,1,fs);

product_derivative = [ecg_derivative; 0].*[0; ecg_derivative];
zero_crossings = product_derivative < 0;
zero_crossings(end) = [];

peaks_index = zero_crossings & (ecg_hfn > threshold);

% determination of the mean heart rate and std (error propagation)
mean_heart_rate = 1/mean(diff(time(peaks_index)))*60;   % per min
std_heart_rate = 1/mean(diff(time(peaks_index)))^2 * ...
                std(diff(time(peaks_index)))*60; %per min

% Plot the data
plot(time,ecg_hfn)
xlabel('Time [sec]');
ylabel('ECG');
title('R-Detection');
hold on
plot(time(peaks_index),ecg_hfn(peaks_index),'og')

dispTxt = sprintf('Heartrate = %.3f ±  %.2f min^{-1}', mean_heart_rate, ...
            std_heart_rate);

text(time(end/2),-2.5,dispTxt);
disp('Done!')    
