% Solution Assignment IIR

% Author: ThH
% Date: Oct-2023
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
impulse_filtered = filter(b, a, impulse);

% define and filter step 
step = zeros(1, duration);
step(t0:end) = 1;
step_filtered = filter(b, a, step);

% Plot the data
ax1 = subplot(211);         % Note that I save the "handle" for the top axis ...
plot(impulse_filtered);
hold('on');
plot(impulse, 'r*')
title('Effects of IIR-Filter');
set(ax1, 'XTickLabel', []);
ylabel('Impulse-Response');

ax2 = subplot(212);         % ... and the handle for bottom axis ...
plot(step_filtered);
hold('on');
plot(step, 'r*')
xlabel('Points')
ylabel('Step-Response')

% "linking" two axes ensures that zooming-in and -out gets done simultaneously
% in both axes
linkaxes([ax1, ax2]);       % ... so that I can "link" them
set(gcf, 'Name', 'Hit any key to continue ...');
shg
pause;
close


%% Part 2

% Get the data
inWorkspace = load('Data/ecg_hfn.mat');
inData = inWorkspace.ecg_hfn;
sample_rate = 1000;

time = 0:1/sample_rate:(length(inData)-1)/sample_rate;

% define filter specs empirically, to eliminate the breathing artifacts
iir_order = 2;
fir_order = 28;
cutoffFreq = 25;
normalizedCutoffFreq = cutoffFreq/(sample_rate/2);

% calculate filter coefficients

[b_fir, a_fir] = fir1(fir_order,   normalizedCutoffFreq, 'low');
[b_iir, a_iir] = butter(iir_order, normalizedCutoffFreq, 'low');

% filter data
EcgFirFiltered = filter(b_fir, a_fir, inData);
EcgIirFiltered = filter(b_iir, a_iir, inData);

% plot data
plot(time, inData)
xlabel('Time (s)');
ylabel('ECG');
xlim([0,  1]);
hold on
plot(time, EcgFirFiltered)
plot(time, EcgIirFiltered)

legend('raw', 'fir', 'iir');
set(gcf, 'Name', 'Hit any key to continue ...');
shg
pause
close;

%% Part 3
% load data
inData = load('data/monitor.mat');
pleth = inData.monitor.signals.pleth;
fs = inData.monitor.param.samplingrate;
L = length(pleth);
time = (0:1:(L-1))/fs;

% get bandpass filter coeffitients 
fCutLow = 1.4;  %Hz
fCutHigh = 5;   %Hz
order = 4;
[b, a] = butter(order, [fCutLow fCutHigh]/(fs/2), 'bandpass');

% filter data
plethFiltered = filter(b, a, pleth);

%plot data
figure()
ax1 = subplot(2, 1, 1);
plot(time, pleth, 'b')
title('Plethysmographie')
ylabel('AU')
ax2 = subplot(2, 1, 2);
plot(time, plethFiltered, 'r')
ylabel('AU')
xlabel('Time (s)')
linkaxes([ax1, ax2], 'xy')
set(gcf, 'Name', 'Hit any key to continue ...');
pause
close('all')

%% Part 4
% Get the data
DicomFile = 'Data/MR-MONO2-16-knee'
data = dicomread(DicomFile);

% Show them
imagesc(data);
title('Original data');
colormap('bone');
colorbar;
set(gcf, 'Name', 'Hit any key to continue ...');
shg
pause

% Find horizontal edges
filtered_h = imfilter(data, fspecial('sobel'));
imagesc(filtered_h);
title('Horizontal Edges');
colorbar;
set(gcf, 'Name', 'Hit any key to continue ...');
shg
pause

% Show positive and negative edges the same color
imagesc(filtered_h.^2);
title('Horizontal Edges - Squared');
colorbar;
set(gcf, 'Name', 'Hit any key to continue ...');
shg
pause

% Find vertical edges, and combine them with the horizontal edges
filtered_v = imfilter(data, fspecial('sobel')');
edges = filtered_h.^2 + filtered_v.^2;

% Show positive and negative edges the same color
imagesc(edges);
title('Edges - Squared');
colorbar;
 