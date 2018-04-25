function [trials, time] = extractAllTrials(data, events, electrode, timeBefore, timeAfter)

% This function extracts all trials of the P300 experiment for one channel
% of EEG.
%
% INPUTS:
%    data      = raw EEG data from Hoffman dataset
%    events    = time of each stimulus from Hoffman dataset
%    electrode = index of the electrode 
%    timeBefore= start of epoch; amount of time before the stimulus
%                appears, in seconds
%    timeAfter = end of epoch; amount of time after the stimulus appears,
%                in seconds
%
% OUTPUTS:
%    trials = matrix of EEG data; every row is one trial, every column is
%             one point in time
%    time   = vector of time points in seconds; can use this to plot trials

fs = 2048; % sampling frequency
ptsBefore = round(timeBefore*fs);  % # of time points before stimulus
ptsAfter = round(timeAfter*fs);    % # of time points after stimulus
time = (-ptsBefore:ptsAfter)/2048;

% Initialize matrix for trial data
trials = zeros(length(events), ptsBefore+ptsAfter+1);

% Remove the zeros from the end of the dataset
datazeros = (sum(data)==0);
lastDataPoint = find(diff(datazeros~=0));
data = data(:,1:lastDataPoint);

% Remove mean from each channel
data = data - repmat(mean(data,2), 1, length(data));

% Re-reference the data
ref = 0.5*(data(7,:) + data(24,:));
data = data - repmat(ref,34,1);

% Extract trial data
for i=1:length(events)
    elapsed = etime(events(i,:), events(1,:));  % elapsed time since trial 1
    ind1 = round(elapsed*fs + 1 + 0.4*fs - ptsBefore);
    ind2 = ind1 + ptsBefore + ptsAfter;
    trials(i,:) = data(electrode, ind1:ind2);
end
