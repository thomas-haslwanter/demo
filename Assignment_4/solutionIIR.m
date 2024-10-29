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
inWorkspace = load('data/ecg_hfn.mat');
inData = inWorkspace.ecg_hfn;
sampleRate = 1000;

time = 0:1/sampleRate:(length(inData)-1)/sampleRate;

% define filter specs empirically, to eliminate the breathing artifacts
iirOrder = 2;
firOrder = 10;    % everything >20 is fine
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
% load data
inData = load('Data/monitor.mat');
pleth = inData.monitor.signals.pleth;
fs = inData.monitor.param.samplingrate;
L = length(pleth);
time = (0:1:(L-1))/fs;
nyq = fs/2; % Nyquist frequency
order = 4;  % order of the filter

% Lowpass-filter
upperFreq = 1.2     % Hz
[b, a] = butter(order, upperFreq/nyq, 'low');
LowpassFiltered = filter(b, a, pleth);

% Highpass-filter
lowerFreq = 1.5    % Hz
[b, a] = butter(order, lowerFreq/nyq, 'high');
HighpassFiltered = filter(b, a, pleth);

% Bandpass filter 
fCutLow = 1.4;      %Hz
fCutHigh = 5;       %Hz
[b, a] = butter(order, [fCutLow fCutHigh]/nyq, 'bandpass');

% filter data
BandpassFiltered = filter(b, a, pleth);

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

DicomFile = 'Data/MR-MONO2-16-knee'
data = dicomread(DicomFile);

% Show them
tiledlayout(2, 2);
ax1 = nexttile;
imagesc(data);
title('Original data');
colormap('bone');
colorbar;


% Find horizontal edges
ax2 = nexttile;
filtered_h = imfilter(data, fspecial('sobel'));
imagesc(filtered_h);
title('Horizontal Edges');
colorbar;

% Show positive and negative edges the same color
ax3 = nexttile;
imagesc(filtered_h.^2);
title('Horizontal Edges - Squared');
colorbar;


% Find vertical edges, and combine them with the horizontal edges
ax4 = nexttile;
filtered_v = imfilter(data, fspecial('sobel')');
edges = filtered_h.^2 + filtered_v.^2;

% Show positive and negative edges the same color
imagesc(edges);
title('H+V-Edges, Squared');
colorbar;

linkaxes([ax1, ax2, ax3, ax4], 'xy');
set(gcf, 'Name', 'Hit any key to continue ...');
shg
 
