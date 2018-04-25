function [time, EEG] = SimEEG_JA(n, fs)
%initialize parameters
Nyquist=fs/2;
NumFreqs=n/2; %for some reason, I'm not including 0?
fVec=linspace(1,Nyquist,NumFreqs);
time=1/fs:1/fs:n/fs;

sig=randn(1,n);
fft_s=fft(sig);
newCoeff=fft_s(1:NumFreqs).*(1./fVec);
newCoeff=[newCoeff, fliplr(newCoeff)];
EEG=abs(ifft(newCoeff));

plot(time,EEG);xlabel('time(s)');ylabel('Power');title('Simulated EEG');
end

