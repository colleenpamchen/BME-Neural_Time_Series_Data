%% Homework1.m 
% Colleen Chen 

% load EEG data 
load 'sub8_sess4_6.mat'
sr = 2048; % sampling rate = 2048 Hz
data = data'; % transpose the data matrix; each column has all the time points per that channel 

nsamps = size(data,1); 
nchans = size(data,2); 

% plot raw EEG data
plot(data)
xlabel('Time (sec)');
ylabel('channelsl'); 
set(gca,'xtick',[])
set(gca,'ytick',[])

% Remove 0's  
row_i = nnz(data(:,1)) % ok, admittedly this is the WORST way to do it... bad logic :( 
ndata = data(1:row_i,:); 
nsamps = size(ndata,1); 
% subtract mean from each channel 
mean_channel = repmat( mean(ndata,1),nsamps,1);
nmdata = ndata - mean_channel; 

figure;
plot(nmdata)
xlabel('Time (sec)');
ylabel('channelsl'); 
set(gca,'xtick',[])
set(gca,'ytick',[])

%%
[trials, time] = extractAllTrials(nmdata,events,elect,t1,t2) ; 

function extractAllTrials(data, events, elec, t1, t2) 




end






