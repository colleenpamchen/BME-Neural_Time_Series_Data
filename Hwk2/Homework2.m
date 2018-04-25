% Homework2 Colleen Chen
close all 
clear 
% Question 1 create test signal composed of high and low frequency sine waves 
Fs = 128;
N = Fs *2; 
time = linspace(-1,1, N); 
% 4 parameters for Amplitude, Frequency, and Phase 
amp = [2 3];
freq = [2 17]; % low and high frequency 
phase = [pi/2 pi/4]; 
sines = zeros(length(amp),length(time) ); 
figure
for i=1:length(amp) 
    sines(i,:)=amp(i).* sin(2*pi*freq(i) .* time + phase(i) );
    subplot(4,1,i); plot(time,sines(i,:))
        xlabel('time(s)');
        ylabel('Amplitude');
        title('Sine wave');   
end
figure
signal = sum( sines + randn(size(sines)) );%sum together and add gaussian noise
plot(time,signal)
    xlabel('time(s)');
    ylabel('Amplitude');
    title('Test signal');

%1B - Create Gaussian Kernel
kernel = gausswin(100); 
kernel = kernel'; 
figure; plot( linspace(-1,1,100), kernel );
    xlabel('time(s)');
    ylabel('Amplitude');
    title('kernel'); 

%1C - convolve signals
convolution = conv(signal, kernel, 'same') / sum(kernel); 
figure; plot(time, convolution, 'r','LineWidth',5)
hold on; plot(time,signal, 'b')
    legend('Convolution', 'Test signal')
    xlabel('Time (s)')
    ylabel('Amplitude');
    title('Convolution'); 

% 1E - take inverse fft
FFTlength = length(signal) + length(kernel)-1; % p138 explains this 
nyq = ceil(FFTlength/2+1); %nyquist frequency 
fft_signal = fft(signal, FFTlength); % FFT of signal
fft_kernel = fft(kernel, FFTlength); % FFT of kernel 
fft_mult = fft_signal.* fft_kernel;
inverse = ifft(fft_mult); 
frequency_bins = linspace(0, Fs/2, nyq ) ; 

figure
subplot(311); plot(frequency_bins, abs(fft_signal(1:nyq))); 
hold on;
plot(frequency_bins, abs(fft_kernel(1:nyq)),'r');
legend('FFT of signal','FFT of kernel')
xlabel('Frequency (Hz)'); ylabel('Magnitude'); 
xlim([0 30])

subplot(312); plot(frequency_bins, abs(fft_mult(1:nyq))); 
xlabel('Frequency (Hz)'); ylabel('Magnitude'); 
xlim([0 30])

subplot(313); plot(time, inverse(1:length(time))/sum(kernel),'r')
hold on; plot(time, signal); 
legend('FFT Result','Test Signal')
xlabel('Time (s)'); 
ylim([-10 10]);

% Question 2: function [time, simEEG] simulateEEG(N, Fs)
[time1, simEEG1]= simulateEEG(1000, 128);
[time2, simEEG2]= simulateEEG(5000, 2048);
[time3, simEEG3]= simulateEEG(2000, 512);
[time4, simEEG4]= simulateEEG(3000, 1024);
figure;
subplot(2,2,1);plot(time1,simEEG1);title('N=1000, Fs=128');
subplot(2,2,2);plot(time2,simEEG2);title('N=5000, Fs=2048');
subplot(2,2,3);plot(time3,simEEG3);title('N=2000, Fs=512');
subplot(2,2,4);plot(time4,simEEG4);title('N=3000, Fs=1024');
xlabel('Time (s)');
% plot Fourier coefficients  
fft_EEG1 = fft(simEEG2); 
figure; loglog( abs( fft_EEG1(1:end/2+1 ))) % plot fourier coefficients on loglog scale
xlabel('Frequency (Hz)');ylabel('Magnitude of Fourier coefficients'); 

% Question 3 create sine waves of 100 hz with variable N
amp = 1;
Fs = 2048;
N = [ 10:10:1000 ];
freq = 100;
sines = cell(1,length(N));
ffts = cell(1,length(N));
frequency_vector = cell(1,length(N));
freq_max = zeros(1,length(N)); 
time_vec = cell(1,length(N));
for i=1:length(N)
    frequency_vector{i} = linspace(0, Fs/2, N(i)/2+1 );
    time_vec{i} = 1/Fs : 1/Fs : N(i)/Fs;
    sines{i} = amp * sin( 2*pi*freq .* time_vec{i} );
    %subplot(4,1,i)
%     figure; plot( time_vec{i}, sines{i} )
    ffts{i} = fft( sines{i} ); 
    ffts{i} = abs( ffts{i}( 1:round(N(i)/2+1) ));
    freq_max(i) = find( ffts{i}==max(ffts{i})); 
    freq_max(i) = frequency_vector{i}(freq_max(i));
end

figure; plot(N, freq_max);
xlabel('Length of N');ylabel('Estimated Frequency (Hz)');

%Question 4a
clear
load sub8_sess4_1.mat
t1 = 0; t2 = 0.4;
Fs = 2048;
nfft = ceil(t2*Fs - t1*Fs);
Nyq = Fs/2;
NumFreqs = round(nfft/2)+1;
freq_vec = linspace(0, Nyq, NumFreqs);
electrode = 1; 
[trials, time] = extractAllTrials(data, events,electrode,t1,t2);
fft_t1 = fft(trials(1,:),nfft);
figure; plot(freq_vec,abs(fft_t1(1:NumFreqs)));
xlabel('Freq(Hz)');ylabel('FFT magnitude');axis([0 1000 0 2000]);
figure; loglog(freq_vec,abs(fft_t1(1:NumFreqs)));
xlabel('Freq(Hz)');ylabel('FFT magnitude');

%4b
TargetTrials = stimuli==6; %logical index for Targets
NonTargetTrials = ~TargetTrials;
TargetsTrials = trials(TargetsTrials,:); %extract Target trials
NonTargetsTrials = trials(NonTargetsTrials,:);
% correction from clasS: mean( abs( fft( data) ))
fft_Targets = abs(fft(TargetsTrials,nfft,2));%fft of Target trials
fft_NonTargets = abs(fft(NonTargetsTrials,nfft,2)); 
MeanTargets = mean(fft_Targets); %then take the mean 
MeanNonTargets=mean(fft_NonTargets);

figure;subplot (121)
loglog(freq_vec, MeanTargets(1:NumFreqs));%target trials
xlabel('Freq(Hz)');ylabel('Power');
hold on
loglog(freq_vec, MeanNonTargets(1:NumFreqs),'r'); %nonTarget
xlabel('Freq(Hz)');ylabel('Power');

subplot(122)
Difference = MeanTargets - MeanNonTargets;
plot(freq_vec, Difference(1:NumFreqs));
xlabel('Freq(Hz)');ylabel('Power');axis([0 200 -500 500]);
title('Difference between Non-target and target trials ')

%4c - rerun above code with new nfft
nfft = 2048;
data_fft = fft(trials', nfft);
freq_vec2 = linspace(0, Fs/2, nfft/2+1);
fft_targets2 = abs( data_fft(1:(nfft/2+1), TargetTrials)); % target trials in cols
fft_nontargets2 = abs( data_fft(1:(nfft/2+1), NonTargetTrials)); % non-target trials

figure; subplot(121); loglog(freq_vec2, mean(fft_targets2,2));
xlabel('Frequency (Hz)'); ylabel('Power')
hold on; loglog(freq_vec2, mean(fft_nontargets2,2),'r')
legend('Target','Non-target')

subplot(122); plot(freq_vec2, mean(fft_nontargets2,2) - mean(fft_targets2,2));
xlabel('Frequency (Hz)'); ylabel('Power')
title('Diff btw Non-target and Target trials')
axis([0 600 -500 500]);

