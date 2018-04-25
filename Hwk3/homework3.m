clear
close all
%1a Family of Morlet Wavelets
n = 4; % number of wavelet cycles  
wfreqs = 4:60; % vector of wavelet frequencies 
fs = 100; 
wtime = -1 : 1/fs : 1; % time vector for wavelet 
    % Standard deviation of the Gaussian: s = n/2pif 
    % Morelet equation: A*exp(-t^2/2s^2) * exp(1i*2*pi*f*t), 
    % where A=1/sqrt(s* sqrt(pi))
wavelet = zeros( length(wfreqs), length(wtime));

for x = 1:length(wfreqs)
    f = wfreqs(x); 
    s = n ./ ( 2* pi .* f );
    A = 1/sqrt(s*sqrt(pi)); % Amplitude changes with parameters 'n' and 'f' 
	wavelet(x,:) = A*exp( -wtime.^2 ./ (2*s^2) ).* exp(1i*2*pi*f.*wtime)  ;
end
realwavelet = real(wavelet); 
figure; plot(wtime,realwavelet);xlabel('Time(s)');ylabel('Amplitude');

%1b - Complex wavelet convolution using the wavelet created just now... 
% test convolution with a 10 Hz sine wave and check against answer 
dfreqs = [10:10:50]; % data of sine waves with these frequencies 
dtime = -5:1/fs:5; % time vector for data 
data = zeros(length(dfreqs),length(dtime)); 
    for k = 1:length(dfreqs)
        data(k,:) = sin( 2*pi*dfreqs(k)*dtime); % test data 
    end
waveletConv = zeros(length(dfreqs),length(wfreqs),length(dtime)); 
% for the test data, loop through the different wavelet frequencies and
% use conv()
figure; 
for k = 1:length(dfreqs)
    for j = 1:length(wfreqs)
        waveletConv(k,j,:) = conv(data(k,:), wavelet(j,:),'same');     
    end
    %plot result of each waveletConv
    subplot(1,5,k); 
    wc = squeeze(abs(waveletConv(k,:,:)));
    imagesc(dtime,wfreqs,wc); 
    set(gca,'ydir','normal'); colormap jet; axis square;
    xlabel('Time (s)');ylabel('Frequency (Hz)'); 
end

%% 1c - rerun part b code with fs_hi = 2048
clear
close all
%1a Family of Morlet Wavelets
n = 4; % number of wavelet cycles  
wfreqs = 4:60; % vector of wavelet frequencies 
fs_hi = 2048;  
wtime = -1 : 1/fs_hi : 1; % time vector for wavelet 
wavelet = zeros( length(wfreqs), length(wtime));
for x = 1:length(wfreqs)
    f = wfreqs(x); 
    s = n ./ ( 2* pi .* f );
    A = 1/sqrt(s*sqrt(pi)); % Amplitude changes with parameters 'n' and 'f' 
	wavelet(x,:) = A*exp( -wtime.^2 ./ (2*s^2) ).* exp(1i*2*pi*f.*wtime)  ;
end
realwavelet = real(wavelet); 
% figure; plot(wtime,realwavelet);xlabel('Time(s)');ylabel('Amplitude');
dfreqs = [10:10:50]; % data of sine waves with these frequencies 
dtime = -5:1/fs_hi:5; % time vector for data 
data = zeros(length(dfreqs),length(dtime)); 
    for k = 1:length(dfreqs)
        data(k,:) = sin( 2*pi*dfreqs(k)*dtime); % test data 
    end
waveletConv = zeros(length(dfreqs),length(wfreqs),length(dtime)); 
    figure; 
for k = 1:length(dfreqs)
    for j = 1:length(wfreqs)
        waveletConv(k,j,:) = conv(data(k,:), wavelet(j,:),'same');     
    end
    wc = squeeze(abs(waveletConv(k,:,:)));
    subplot(1,5,k); 
    imagesc(dtime,wfreqs,wc); 
    set(gca,'ydir','normal'); colormap jet; axis square; 
    xlabel('Time (s)');ylabel('Frequency (Hz)'); 
end

% 1c increase the test frequency range 
%n = 4; % number of wavelet cycles  
n=8; 
fs_hi = 2048;  
wtime = -1 : 1/fs_hi : 1; % time vector for wavelet 
wfreqs = 10:10:1000; % INCREASE WFREQS RANGE  
wavelet = zeros( length(wfreqs), length(wtime));
for x = 1:length(wfreqs)
    f = wfreqs(x); 
    s = n ./ ( 2* pi .* f );
    A = 1/sqrt(s*sqrt(pi)); % Amplitude changes with parameters 'n' and 'f' 
	wavelet(x,:) = A*exp( -wtime.^2 ./ (2*s^2) ).* exp(1i*2*pi*f.*wtime)  ;
end
realwavelet = real(wavelet); 
% figure; plot(wtime,realwavelet);xlabel('Time(s)');ylabel('Amplitude');
dfreqs = [600:100:1000]; % INCREASE DFREQS RANGE, but end up with only 5 plots.  
dtime = -5:1/fs_hi:5; % time vector for data 
data = zeros(length(dfreqs),length(dtime)); 
    for k = 1:length(dfreqs)
        data(k,:) = sin( 2*pi*dfreqs(k)*dtime); % test data 
    end
waveletConv = zeros(length(dfreqs),length(wfreqs),length(dtime)); 
figure; 
for k = 1:length(dfreqs)
    for j = 1:length(wfreqs)
        waveletConv(k,j,:) = conv(data(k,:), wavelet(j,:),'same');     
    end
    subplot(1,5,k); 
    wc = squeeze(abs(waveletConv(k,:,:)));
    imagesc(dtime,wfreqs,wc); 
    set(gca,'ydir','normal'); axis square; colormap jet 
    xlabel('Time (s)');ylabel('Frequency (Hz)'); 
    title(sprintf('%d',dfreqs(k)))
end

%% 1d Temporal resolution 
clear 
wfreqs = 4:60; 
fs_hi = 2048; 
wtime = -1 : 1/fs_hi : 1; % time vector for wavelet 
wavelet = zeros( length(wfreqs), length(wtime));
        dtime = -5: 1/fs_hi :5; % time vector for data 
        thalf = ceil(length(dtime)/2); 
        data = zeros(1,length(dtime)); 
        data(1:thalf) = sin( 2*pi*10*dtime(1:thalf) );
        data(thalf+1:end) = sin( 2*pi*20*dtime(thalf+1:end) ); 
waveletConv = zeros(length(wfreqs),length(dtime));             
n = [3 10]; 
figure; 
for nn = 1:length(n)
        for x = 1:length(wfreqs)
                f = wfreqs(x); 
                s = n(nn) ./ ( 2* pi .* f );
                A = 1/sqrt(s*sqrt(pi)); % Amplitude changes with parameters 'n' and 'f' 
                wavelet(x,:) = A*exp( -wtime.^2 ./ (2*s^2) ).* exp(1i*2*pi*f.*wtime)  ;
            waveletConv(x,:) = conv(data, wavelet(x,:),'same'); 
        end
            %plot result of waveletConv
                subplot(1,3,nn);imagesc(dtime, wfreqs, abs(waveletConv)); %looking at frequency bands
                xlabel('Time (s)'); ylabel('Frequency (Hz)')
                set(gca,'ydir','normal'); colormap jet ; axis square;
                title(sprintf('n = %d',n(nn)) )             
                subplot(133); hold on; %looking at time domain 
      plot(dtime,abs(waveletConv(7,:))); xlabel('time(s)');ylabel('magnitude'); axis([-.5 .5 0 1000]); axis square;
end


%% 2 - pre-process P300 data
clear
load sub8_sess4_6;
Cz = data(32,:); %extract channel 32
ref = mean( data([7 24],:) ); %the reference, taken from 7 and 24 
Cz = Cz-ref; %re-referencing Cz to Ref
Cz = Cz(1:104448); %remove zeros 
DC = mean(Cz); 
Cz = Cz - repmat(DC,1,length(Cz)); %remove DC 
fs=2048;
time = (1:length(Cz))/fs ;
figure; plot(time,Cz);xlabel('time(s)');ylabel('Magnitude');
title('Cz channel 32 EEG')

%2b - wavelet conv
wfreqs = 4:60;
n = 3; %this parameter was good for temporal precision 
wtime = -1: 1/fs :1;
wavelet = zeros(length(wfreqs),length(wtime));
waveletConv = zeros(length(wfreqs),length(Cz)); 
for x = 1:length(wfreqs)
                f = wfreqs(x); 
                s = n ./ ( 2* pi .* f );
                A = 1/sqrt(s*sqrt(pi)); % Amplitude changes with parameters 'n' and 'f' 
                wavelet(x,:) = A*exp( -wtime.^2 ./ (2*s^2) ).* exp(1i*2*pi*f.*wtime)  ;
                waveletConv(x,:) = conv(Cz', wavelet(x,:),'same');            
end
%     figure; plot(wtime,wavelet(1,:))
    figure; imagesc(time,wfreqs,abs(waveletConv)); 
    axis([0 4 4 60]); set(gca,'ydir','normal'); colorbar; colormap jet; 
    set(gca,'clim',([0 5000]));xlabel('time(s)');ylabel('Freq(Hz)'); 

%% 2c - extract target vs non-target trials 
t1 = 0; t2 = 0.4;
ptsBefore = round(t1*fs); % # of time points before stimulus
ptsAfter = round(t2*fs); % # of time points after stimulus
time = (ptsBefore:ptsAfter)/2048;
targetTrials = zeros(length(wfreqs), length(time) );
nontargetTrials = zeros(length(wfreqs), length(time) );
% Calculate mean of target and non-target trials using the etime index to
% extract trials 
targetindex = find(stimuli==target); 
nontargetindex = find( ~(stimuli==target)); 
    for i = 1:length(targetindex)
         elapsed = etime(events(i,:), events(1,:)); % elapsed time since trial 1
         ind1 = round(elapsed*fs + 1 + 0.4*fs - ptsBefore);
         ind2 = ind1 + ptsBefore + ptsAfter;
         targetTrials = targetTrials + abs(waveletConv(:, ind1:ind2));    
    end
targetTrials = targetTrials/length(targetindex);
    for j = 1: length(nontargetindex)
     elapsed = etime(events(j,:), events(1,:)); % elapsed time since trial 1
     ind1 = round(elapsed*fs + 1 + 0.4*fs - ptsBefore);
     ind2 = ind1 + ptsBefore + ptsAfter;
     nontargetTrials = nontargetTrials + abs(waveletConv(:, ind1:ind2));
    end
nontargetTrials = nontargetTrials/length(nontargetindex);

figure; subplot(121)
imagesc(time, wfreqs, targetTrials); colorbar; colormap jet; axis square  
set(gca,'ydir','normal', 'clim',[700 3000] )
xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('Target trials')
subplot(122)
imagesc(time, wfreqs, nontargetTrials); colorbar; colormap jet; axis square 
set(gca,'ydir','normal', 'clim',[700 3000])
xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('Non-target trials')

