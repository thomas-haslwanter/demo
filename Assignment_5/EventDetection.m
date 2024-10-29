%EventDetection Show how events can be elegantly detected using binary indexing
%***********************************************

% author:  Thomas Haslwanter
% date:    Nov-2023

% Get eye positions, sampled with 100 Hz
load HorPos;	% This file has to exist in your current directory!
rate = 100;

% Select an interesting domain
myDomain = 9000 : 13500;

% Plot 1: raw data
ax1 = subplot(3,2,1)
plot(HorPos(myDomain))
ylabel('Eye Position [deg]');
axis tight

% Plot 2: absolute velocity
orderPoly = 3;
winSize = 71;
deriv = 1;
eyeVelocity = savgol(HorPos, orderPoly, winSize, deriv, rate);
eyeAbsoluteVelocity = abs(eyeVelocity);

ax2 = subplot(3,2,3)
plot(eyeAbsoluteVelocity(myDomain))
ylabel('Absolute Eye Velocity [deg]')
axis tight

% Set a default threshold, in case the threshold is not determined
% interactively
threshold = 6.3;

%To find the threshold interactively, use the following lines
% set(gcf, 'Name', 'Select the threshold:')
% selectedPoint = ginput(1);
% threshold = selectedPoint(2);    % I only want the y-value
% set(gcf, 'Name', '');
line(xlim, [threshold threshold], 'Color', 'r')

% Plot3: show where the absolute velocity exceeds the threshold
isFast = eyeAbsoluteVelocity > threshold;

ax3 = subplot(3,2,5)
plot(isFast(myDomain), '-x')
axis tight
ylabel('Above threshold')

% Plot4: as Plot3, but zoomed in
closeDomain = 9900 : 10600;

ax4 = subplot(3,2,2)
plot(isFast(closeDomain), '-x')
axis tight
ylabel('Above threshold');

% Plot5: Find the start and end of each movement
ax5 = subplot(3,2,4)
startStop = diff(isFast);
plot(startStop(closeDomain))
ylabel('Start / Stop')

linkaxes([ax1 ax2 ax3], 'x');
linkaxes([ax4, ax5], 'x');

% Find the start and end times for all movements (in sec)
movementStartTimes = find(startStop ==  1)/rate
movementEndTimes   = find(startStop == -1)/rate


