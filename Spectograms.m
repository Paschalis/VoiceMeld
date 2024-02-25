% Load the recorded audio file
[audioData, fs] = audioread('recorded_audio.wav');

% Store the recording duration in seconds
recordingDuration = 3;

% Calculate the length of the signal
%L = recordingDuration*fs;
L = size(audioData,1);

% Fast Fourier Transform before spectogram creation
figure(1);
Y = fft(audioData);
plot(fs/L*(0:L-1),abs(Y),'LineWidth',3)
title('Complex Magnitude of fft Spectrum')
xlabel('f (Hz)')
ylabel('|fft(Audio)|')


figure(2);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs/L*(0:(L/2));
plot(f,P1,'LineWidth',3)
title('Single-Sided Amplitude Spectrum of audioData(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% Parameters for the spectrogram
window10ms = hamming(round(fs * 0.01),'symmetric'); % 10 msec Hamming window
window100ms = hamming(round(fs * 0.1),'symmetric'); % 100 msec Hamming window
overlap = round(fs * 0.005); % 5 msec overlap

% Spectrogram with 10 msec Hamming window and 5 msec overlap
figure(3);
subplot(2,1,1);
spectrogram(audioData, window10ms, overlap, [], fs, 'yaxis');
title('Spectrogram with 10ms Hamming Window and 5ms Overlap');
xlabel('Time (s)');
ylabel('Frequency (kHz)');
subplot(2,1,2);
% Spectrogram with 100 msec Hamming window and 5 msec overlap
spectrogram(audioData, window100ms, overlap, [], fs, 'yaxis');
title('Spectrogram with 100ms Hamming Window and 5ms Overlap');
xlabel('Time (s)');
ylabel('Frequency (kHz)');
