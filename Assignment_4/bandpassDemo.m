%Demonstration of a 5th order Butterworth bandpass filter
%
% Inputs
%   inData : data to be filtered
%   rate   : sample rate [Hz]
%   lowFreq:  lower frequency of bandpass [Hz]
%   highFreq: upper frequency of bandpass [Hz]
%
% Returns:
%   filtered : filtered data
%
%
% Note:
%   If the function is called without arguments, a test signal with
%   two frequencies is generated; the selected bandpass only lets
%   the higher frequency through.
%
% Examples:
%   bandpass();
%   
% author: ThH
% date:   oct-2015
% ver:    0.1
%*****************************************************************

function filtered = bandpassDemo(inData, rate, lowFreq, highFreq)

%% Test-signal
if nargin == 0
    % Generate test-data
    testFlag = 1;
    rate = 1000;

    t = 0:1/rate:2;
    
    f1 = 10;
    f2 = 40;
    inData = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t);
    
    % Set the default filter parameters
    lowFreq = 25;
    highFreq = 75;
else
    testFlag = 0;
end

nyq = rate/2;   % Nyquist frequency
    
%% Bandpass filter
% Note that these two lines are the only ones really needed!

[b, a] = butter(5,[lowFreq highFreq]/nyq, 'bandpass');
filtered = filter(b, a, inData);

%% Show the test-signal
if testFlag == 1
    plot(t, inData);
    hold('on');
    plot(t, filtered, 'r');
    legend('rawData', 'filtered');
end

