% Solution Assignment IIR

% Author: ThH
% Date: Nov-2023
%*****************************************************************

%% Part 1
% Clear away previous inputs
clear('all'), close('all'), clc

% define filter coefficients
a = [1, -0.5095];
b =[ 0.7548, -0.7548];
duration = 100;
t0 = 10;

% define and filter impulse
impulse = zeros(1, duration);
impulse(t0) = 1;
impulseFiltered = filter(b, a, impulse);

% define and filter step 
step = zeros(1, duration);
step(t0:end) = 1;
stepFiltered = filter(b, a, step);

% Plot the data
tiledlayout('vertical')
ax1 = nexttile;         % Note that I save the "handle" for the top axis ...
plot(impulseFiltered);
hold('on');
plot(impulse, 'r*')
title('Effects of IIR-Filter');
set(ax1, 'XTickLabel', []);
ylabel('Impulse-Response');

ax2 = nexttile;         % ... and the handle for bottom axis ...
plot(stepFiltered);
hold('on');
plot(step, 'r*')
xlabel('Points')
ylabel('Step-Response')

% "linking" two axes ensures that zooming-in and -out gets done simultaneously
% in both axes
linkaxes([ax1, ax2], 'x');       % ... so that I can "link" them
set(gcf, 'Name', 'Hit any key to continue ...');
shg
pause;
close


%% Part 2

% Get the data
inWorkspace = load('Data/ecg_hfn.mat');
inData = inWorkspace.ecg_hfn;
sampleRate = 1000;

time = 0:1/sampleRate:(length(inData)-1)/sampleRate;

% define filter specs empirically, to eliminate the breathing artifacts
iirOrder = 2;
firOrder = 20;    % everything >20 is fine
cutoffFreq = 25;
normalizedCutoffFreq = cutoffFreq/(sampleRate/2);

% calculate filter coefficients

[b_fir, a_fir] = fir1(firOrder,   normalizedCutoffFreq, 'low');
[b_iir, a_iir] = butter(iirOrder, normalizedCutoffFreq, 'low');

% filter data
EcgFIRFiltered = filter(b_fir, a_fir, inData);
EcgIIRFiltered = filter(b_iir, a_iir, inData);

% plot data
plot(time, inData)
xlabel('Time (s)');
ylabel('ECG');
xlim([0,  1]);
hold on
plot(time, EcgFIRFiltered)
plot(time, EcgIIRFiltered)

legend('Original Data', 'FIR', 'IIR');
set(gcf, 'Name', 'Hit any key to continue ...');
shg
pause
close;

%% Part 3

    % [xxx ----------------- To-do ---------------------------------
    %  For the plots you have to generate the following variables:
    %    * time
    %    * pleth (original data)
    %    * LowpassFiltered (lowpass)
    %    * HighpassFiltered (highpass)
    %    * BandpassFiltered (bandpass)
    %  ------------------------------------------------------------- xxx]

%plot data
figure()
tiledlayout("vertical");
ax1 = nexttile;
plot(time, pleth)
title('Plethysmographie')
ylabel('Measurement')
xticklabels = '';

ax2 = nexttile;
plot(time, LowpassFiltered)
ylabel({'Lowpass', '"Breathing"'})
xticklabels = '';

ax3 = nexttile
plot(time, HighpassFiltered)
ylabel({'Highpass', '"Heartbeat"'})
xlabel('Time (s)')

linkaxes([ax1, ax2, ax3], 'x')
set(gcf, 'Name', 'Hit any key to continue ...');
pause
close('all')

%% Part 4
% Get the data
    % [xxx ----------------- To-do ---------------------------- xxx]

