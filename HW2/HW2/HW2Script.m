close all
clear
%Question 1
%Create sine waves with arbitrary parameters
amp=[2 3 3 4];
sr=1000;
time=0:1/sr:2;
freq=[2 5 10 17];
phase=[pi/2 pi/4 pi/8 pi/16]
for i=1:4
    sines(i,:)=amp(i).*sin(2*pi*freq(i).*time + phase(i));
    subplot(4,1,i)
    plot(time,sines(i,:))
end
figure
ComSines=sum(sines+randn(size(sines)));%sum together and add noise
plot(time,ComSines)

%1b - Create Gaussian Kernel
kernel=gausswin(101); kernel=kernel'
figure
subplot(2,1,1);plot(time,ComSines);title('summed sine waves');
subplot(2,1,2);plot((1:length(kernel))./1000,kernel);xlabel('time(s)');axis([0 2 0 1]);title('kernel'); %plot kernel below summed sines

%1c - convolve signals
figure
convsig=conv(ComSines,kernel,'same'); convsig=convsig/sum(kernel);
plot(time,convsig,'LineWidth',5);;hold on;plot(time,ComSines,'r','linewidth',.5);legend(gca,'Convolved','Original')

%1e - take inverse fft
FFTlength=length(ComSines)+length(kernel)-1;
fft_ComSines=fft(ComSines,FFTlength);
fft_kernel=fft(kernel,FFTlength);
fft_mult=fft_ComSines.*fft_kernel;
inverse=ifft(fft_mult); inverse=inverse/sum(kernel);
inverse=inverse(ceil(length(kernel)/2):end-floor(length(kernel)/2));
subplot(221);plot(fft_ComSines);title('FFT of test signal');
subplot(222);plot(fft_kernel);title('FFT of kernel');
subplot(223);plot(fft_mult);title('multiply spectra together');
subplot(224);plot(time,inverse);title('inverse fourier');

%Question 2- Plot simulated EEGs
[time1, EEG1]=SimEEG(5000,500);[time2,EEG2]=SimEEG(1000,250);[time3,EEG3]=SimEEG(3000,1000);[time4,EEG4]=SimEEG(5000,2048);
subplot(2,2,1);plot(time1,EEG1)
subplot(2,2,2);plot(time2,EEG2)
subplot(2,2,3);plot(time3,EEG3)
subplot(2,2,4);plot(time4,EEG4)

fft_EEG1=fft(EEG1); loglog(abs(fft_EEG1(1:end/2+1))) %plot fourier coefficients on loglog scale

%3a - create sine waves of 100 hz with variable n
amp=1;
sr=2048;
n=[100:3000];
freq=100;
sines=cell(1,length(n));
ffts=cell(1,length(n));
fVecs=cell(1,length(n));
times=cell(1,length(n));
for i=1:length(n)
    fVecs{i}=linspace(0,sr/2,round(n(i)/2+1));
    times{i}=1/sr:1/sr:n(i)/sr;
    sines{i}=amp*sin(2*pi*freq.*times{i});
    %subplot(4,1,i)
    plot(times{i},sines{i})
    ffts{i}=fft(sines{i}); ffts{i}=abs(ffts{i}(1:round(n(i)/2+1)));
    EstFreq(i)=find(ffts{i}==max(ffts{i})); EstFreq(i)=fVecs{i}(EstFreq(i));
end

figure;plot(n,EstFreq);xlabel('Number Samples');ylabel('Freq(Hz)');

%Question 4
clear
load sub8_sess4_1.mat
before=0;after=.4;
fs=2048;
nfft=ceil(after*fs-before*fs);
Nyquist=fs/2;
NumFreqs=round(nfft/2)+1;
fVec=linspace(0,Nyquist,NumFreqs);
[trials time]=extractAllTrials(data,events,1,before,after);
fft_t1=fft(trials(1,:),nfft);
plot(fVec,abs(fft_t1(1:NumFreqs)));xlabel('Freq(Hz)');ylabel('Power');axis([0 500 0 2000]);
loglog(fVec,abs(fft_t1(1:NumFreqs)));xlabel('Freq(Hz)');ylabel('Power');

%4b
TargetTrials=stimuli==6;NonTargetTrials=~TargetTrials;
TargetTrials=trials(TargetTrials,:);NonTargetTrials=trials(NonTargetTrials,:);
fft_Targets=abs(fft(TargetTrials,nfft,2));fft_NonTargets=abs(fft(NonTargetTrials,nfft,2)); 
MeanTargets=mean(fft_Targets);MeanNonTargets=mean(fft_NonTargets);

subplot (211)
loglog(fVec,MeanTargets(1:NumFreqs));xlabel('Freq(Hz)');ylabel('Power');
hold on
loglog(fVec,MeanNonTargets(1:NumFreqs),'r');xlabel('Freq(Hz)');ylabel('Power');

subplot(212)
Difference=MeanTargets-MeanNonTargets;
plot(fVec,Difference(1:NumFreqs));xlabel('Freq(Hz)');ylabel('Power');axis([0 200 -500 500]);

temp=abs(Difference);
MaxDiff=find(temp==max(temp));
MaxDiff=find(temp==max(temp(2:NumFreqs)));
MaxDiff=find(temp==max(temp(3:NumFreqs)));
%4c - rerun above code with new nfft
nfft=2048;



