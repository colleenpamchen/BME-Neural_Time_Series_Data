function [trials, time] = extractAllTrials(data, events,  t1, t2)
% modified to extract trials from EEG data from one channel that has
% already been preprocessed. CC 
fs = 2048; % sampling frequency
ptsBefore = round(t1*fs);  % # of time points before stimulus
ptsAfter = round(t2*fs);    % # of time points after stimulus
time = (-ptsBefore:ptsAfter)/2048;

% Initialize matrix for trial data
trials = zeros(length(events), ptsBefore+ptsAfter+1);

% Extract trial data
for i=1:length(events)
    elapsed = etime(events(i,:), events(1,:));  % elapsed time since trial 1
    ind1 = round(elapsed*fs + 1 + 0.4*fs - ptsBefore);
    ind2 = ind1 + ptsBefore + ptsAfter;
    trials(i,:) = data(:, ind1:ind2);
end


