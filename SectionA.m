clear all; clc; close all;
% Load the recorded audio file
[audioData, fs] = audioread('recorded_audio_1.wav');
recordingDuration=3;

desired_fs = 16000; 
% Set the target sampling rate
targetFs = desired_fs; % Replace with your desired target sampling rate

% Perform resampling
resampledAudio = audioData;

% Plot the original and resampled waveforms
timeOriginal = (0:length(audioData)-1) /targetFs;
timeResampled = (0:length(resampledAudio)-1) / targetFs;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Design a 101-point FIR equiripple highpass filter
filterOrder = 101;
cutOffFrequency = 50; % Adjust this based on your specific requirements
firHighpassFilter = designfilt('highpassfir', 'FilterOrder', filterOrder, 'CutoffFrequency', cutOffFrequency, 'SampleRate', targetFs);

% Apply the highpass filter to the resampled audio
filteredAudio = filter(firHighpassFilter, resampledAudio);

% Choose the waveform to be used
desiredAudio = audioData;

% Play the filtered recorded audio
%sound(filteredAudio, desired_fs);

%L = 640; % 40 msec frame duration
L = 480; % 30 msec frame duration
R = 160; % 10 msec frame shift

% Parameters for short-time analysis
frameSize = L; % 300 Samples at Fs=10kHz
frameOverlap = L - R; % 100 Samples at Fs=10kHz


% A3
% Use buffer for overlapping frames
bufferedAudio = buffer(desiredAudio, L, frameOverlap);

% Get the number of frames
numFrames = size(bufferedAudio, 2);
%Number of Frames = (Signal Duration - Frame Duration) / (Frame Shift - Frame Overlap) + 1
numFramesCrossCheck = floor(abs(((length(desiredAudio) - L) / (R - frameOverlap)) + 1));

% Initialize arrays for short-time log energy and zero crossing rate
logEnergy = zeros(1, numFrames);
Energy = zeros(1, numFrames);
zeroCrossingRate = zeros(1, numFrames);

% Compute short-time log energy and zero crossing rate for each frame
for i = 1:numFrames
    currentFrame = bufferedAudio(:, i);
    
    % Apply Hamming window to the frame
    hammingWindow = hamming(frameSize,'symmetric');
    %hammingWindow = hamming(frameSize,'periodic');
    windowedFrame = currentFrame .* hammingWindow;
    
    % Short-time log energy
    logEnergy(i) = 10 * log10(sum(windowedFrame.^2));
    Energy(i) = sum(windowedFrame.^2);
    
    % Short-time zero crossing rate
    zeroCrossingRate(i) = R * sum(abs(diff(sign(windowedFrame)))) / (2 * frameSize);
end
% Normalized energy at 0
normalizedLogEnergy = logEnergy - max(logEnergy);

% Plot the short-time log energy and zero crossing rate
frameTime = (0:numFrames-1) * frameOverlap / targetFs;

%IZCT = max(35, mean(zeroCrossingRate(1:3)) + 3 * std(zeroCrossingRate(1:3)));
IZCT = 35;
ITU = -12; % A constant between -10 and -20 ~-12
ITR = -max(-(ITU-10), (mean(logEnergy(1:3)) + 3 * std(logEnergy(1:3))));
%ITR = ITU-10;



figure(8);
subplot(4,1,1);
plot(timeOriginal, desiredAudio);
title('Speech Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(4,1,2);
plot(1:numFrames, normalizedLogEnergy);
title('Short-Time Log Energy with Hamming Window');
xlabel('Frame Number');
ylabel('Log Energy');
legend('Normalized Log Energy','ITU' , 'ITR')

subplot(4,1,3);
plot(1:numFrames, zeroCrossingRate);
title('Short-Time Zero Crossing Rate with Hamming Window');
xlabel('Frame Number');
ylabel('Zero Crossing Rate Zc');
legend('Short-Time Zero Crossing Rate','IZCT' )




%Define the number of words contained in the wav file
numOfWords=1;
%2 iterations process of E and B values calculation because the wav file

initialFrame = 1;
framesHorizon = numFrames;
check=1;

% Step 1: Forward search to find B1
while initialFrame <= framesHorizon
    % Find the first frame where log energy exceeds lower threshold (ITR)
    if normalizedLogEnergy(initialFrame) > ITR
        % Check the region around the detected frame
        regionCheck = normalizedLogEnergy(initialFrame-1:initialFrame+1) > ITU;
        if all(regionCheck)
            % Stable initial frame found
            B1 = initialFrame;
            break;
        end
    end
    initialFrame = initialFrame + 1;
end

% Step 2: Backward search to find E1
initialFrame = framesHorizon;
while initialFrame >= 1
    % Find the first frame where log energy exceeds lower threshold (ITR)
    if normalizedLogEnergy(initialFrame) > ITR
        % Check the region around the detected frame
        regionCheck = normalizedLogEnergy(initialFrame-1:initialFrame+1) > ITU;
        if all(regionCheck)
            % Stable initial frame found
            E1 = initialFrame;
            break;
        end
    end
    initialFrame = initialFrame - 1;
end

% Step 3: Search backward from B1 to B1-25 for zero-crossing count
if sum(zeroCrossingRate(B1-25:B1) > IZCT) >= 4
    for i=B1-25:B1
        if zeroCrossingRate(i) > IZCT
            B2 = i;
            break; 
        end
    end   
else
    B2 = B1;
end


% Step 4: Search forward from E1 to E1+25 for zero-crossing count
if sum(zeroCrossingRate(E1:E1+25) > IZCT) >= 4
    for i=E1:E1+25
        if zeroCrossingRate(i) > IZCT
            E2 = i;
            break;
        end
    end   
else
    E2 = E1;
end


% Step 5: Final check around [B2, E2] for log energy threshold (ITR)
% Modify beginning and/or ending frame to match the extended region 
if check==1
    for i = B2-25:B2
        if normalizedLogEnergy(i) > ITR 
            temporaryExceedingFrames=find(normalizedLogEnergy > ITR);
            B2 = temporaryExceedingFrames(1)-1;
            break;
        end
    end
    for i = E2:E2+25
        if normalizedLogEnergy(i) > ITR       
            temporaryExceedingFrames=find(normalizedLogEnergy > ITR);
            E2 = temporaryExceedingFrames(end)+1;
            break;
        end
    end
end



% Plot lines for beginning and ending frames on the log energy plot
subplot(4,1,2);
hold on;
line(get(gca, 'XLim'), [ITU, ITU], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
line(get(gca, 'XLim'), [ITR, ITR], 'Color', 'c', 'LineStyle', '--', 'LineWidth', 1.5);
line([B1, B1], get(gca, 'YLim'), 'Color', 'g', 'LineStyle', '--', 'LineWidth', 1.5);
line([E1, E1], get(gca, 'YLim'), 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
line([B2, B2], get(gca, 'YLim'), 'Color', 'm', 'LineStyle', '--', 'LineWidth', 1.5);
line([E2, E2], get(gca, 'YLim'), 'Color', 'b', 'LineStyle', '--', 'LineWidth', 1.5);
hold off;
title('Short-Time Log Energy with Hamming Window');
xlabel('Frame Number');
ylabel('Log Energy');
legend('Normalized Log Energy','ITU' , 'ITR');
subplot(4,1,3);
plot(1:numFrames, zeroCrossingRate);
hold on;
line(get(gca, 'XLim'), [IZCT, IZCT], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.5);
line([B1, B1], get(gca, 'YLim'), 'Color', 'g', 'LineStyle', '--', 'LineWidth', 1.5);
line([E1, E1], get(gca, 'YLim'), 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.5);
line([B2, B2], get(gca, 'YLim'), 'Color', 'm', 'LineStyle', '--', 'LineWidth', 1.5);
line([E2, E2], get(gca, 'YLim'), 'Color', 'b', 'LineStyle', '--', 'LineWidth', 1.5);
hold off;
title('Short-Time Zero Crossing Rate with Hamming Window');
xlabel('Frame Number');
ylabel('Zero Crossing Rate Zc');
legend('Short-Time Zero Crossing Rate','IZCT' , 'B1', 'E1' , 'B2', 'E2')




% Define the section of our interest
% Focus on the voiced unvoiced section
notSilencedzeroCrossingRate = zeroCrossingRate(B2:E2);
unvoicedAudionSectionIndeces = find(notSilencedzeroCrossingRate>IZCT);
% Find voicedAudionSectionFrames and unvoicedAudionSectionIndeces
unvoicedAudionSectionFrames = unvoicedAudionSectionIndeces + (B2-1);
voicedAudionSectionFrames = zeros(1,length(notSilencedzeroCrossingRate)-length(unvoicedAudionSectionFrames));
voicedIndex=1;
for i=1:length(notSilencedzeroCrossingRate)
    check=0;
    for j=1:length(unvoicedAudionSectionIndeces)
        if unvoicedAudionSectionIndeces(j) == i
            check = check+1;
        end
    end
    if check == 0 
        voicedAudionSectionFrames(voicedIndex)=i+(B2-1);
        voicedIndex = voicedIndex+1;
    end
end


subplot(4,1,2);
hold on;
% Highlight voiced frames in green
scatter(voicedAudionSectionFrames, -95*ones(1,length(voicedAudionSectionFrames)), 'g', 'filled');
% Highlight unvoiced frames in cyan
scatter(unvoicedAudionSectionFrames, -95*ones(1,length(unvoicedAudionSectionFrames)), 'c', 'filled');
% Highlight unvoiced frames in red
scatter([1:B2-1,E2+1:numFrames], -95*ones(1,length([1:B2-1,E2+1:numFrames])), 'r', 'filled');
hold off;

subplot(4,1,3);
hold on;
% Highlight voiced frames in green
scatter(voicedAudionSectionFrames, 0*ones(1,length(voicedAudionSectionFrames)), 'g', 'filled');
% Highlight unvoiced frames in cyan
scatter(unvoicedAudionSectionFrames, 0*ones(1,length(unvoicedAudionSectionFrames)), 'c', 'filled');
% Highlight unvoiced frames in red
scatter([1:B2-1,E2+1:numFrames], 0*ones(1,length([1:B2-1,E2+1:numFrames])), 'r', 'filled');
hold off;



% A4


% Step1: Extract the current window -Divide the voice waveform into overlapping frames. 
% Apply a window function to each frame to reduce spectral leakage. 
% Frame segmentation - L changed because we need  30 msec
% Has been already executed - bufferedAudio

% Initialize pitch_periods array 
pitch_periods = zeros(1, size(bufferedAudio, 2));
freq = pitch_periods;
autocorrValues = zeros(2*L-1, size(bufferedAudio, 2));
lags = zeros(2*L-1, size(bufferedAudio, 2));

% Iterate through frames 
for i = 1:size(bufferedAudio, 2)
    % Step2: Compute the short-time autocorrelation using the provided
    % equation - Calculate the autocorrelation function for each frame. 
    % This involves computing the correlation of the frame with delayed 
    % versions of itself over different time lags.
    % Define the analysis window (e.g., Hamming window)
    
    % Step 3: Autocorrelation calculation
    % Autocorrelation calculation for the purpose of calculating the 
    % fundamental frequencies - applying Hamming windows 
    hammingWindow = hamming(L,'symmetric');
    %[autocorrValues(:,i),lags(:,i)] = xcorr(bufferedAudio(:,i).*hammingWindow, 'coeff'); 
    [autocorrValues(:,i),lags(:,i)] = xcorr(bufferedAudio(:,i), 'coeff');
    
    % Process of locating the second peak using a temporary variable -
    % making all the autocorralation values around first peak 
    
    % Step 4: Peak Picking
    % Identify peaks in the autocorrelation function. 
    % Peaks at non-zero lags represent repeating patterns or 
    % periodicity in the signal.
    rxx = autocorrValues(:,i);
    findNegValsrxx = find(rxx < 0);
    rxx(findNegValsrxx) = 0;
    centerPeakWidth = find(rxx(L:end)==0,1);
    rxx(L-centerPeakWidth+1 : L+centerPeakWidth) = min(rxx);

    % Find the lag corresponding to the maximum autocorrelation 
    [max_value, max_index] = max(rxx);

    % Step 5 : Pitch Period Extraction:
    % Determine the lag corresponding to the highest peak in the 
    % autocorrelation function. This lag represents the pitch period of 
    % the frame. Convert lag to pitch period 
    pitch_periods(i) = (abs(max_index - L))*(L/fs);
    freq(i) =  fs / pitch_periods(i) ;  
end

% Apply pitch to voiced segments
pitchPeriods = pitch_periods;
pitchPeriods(unvoicedAudionSectionFrames)=0;
pitchPeriods([1:B2,E2:numFrames])=0;
frequencies = freq;
frequencies(unvoicedAudionSectionFrames)=0;
frequencies([1:B2,E2:numFrames])=0;


figure(8);

subplot(4,1,4);
%plot(recordingDuration*(0:length(pitchValues) - 1)/(length(pitchValues) - 1), pitchValues);
stem(voicedAudionSectionFrames,frequencies(voicedAudionSectionFrames));
hold on;
plot(frequencies,'r');
hold off;
title('Fundamental Frequency (Pitch) in Hz');
xlabel('Frames');
ylabel('Pitch Values [Hz]');





%A5

% Every frames has a duration of 30msec, the first unvoiced and the last
% voiced frames have been chosen
chosenVoiced = voicedAudionSectionFrames(end);
chosenUnvoiced = unvoicedAudionSectionFrames(1);
j=0;
Hfilter=0;
Hfilter8=0;
Hfilter12=0;
Hfilter16=0;
for p = [8,12,16]
    Hfilter=0;
    j = j+1;
    windowSize = 5; 
    %b = (1/windowSize)*ones(1,windowSize);
    a1 = lpc(bufferedAudio(:,chosenVoiced),p);
    a2 = lpc(bufferedAudio(:,chosenUnvoiced),p);
    est_xv = filter([0 -a1(2:end)],1,bufferedAudio(:,chosenVoiced));
    est_xu = filter([0 -a2(2:end)],1,bufferedAudio(:,chosenUnvoiced));
    
    figure(14);
    subplot(3,1,j);
    plot(1:L,bufferedAudio(:,chosenVoiced),1:L,est_xv,'--',1:L,abs(bufferedAudio(:,chosenVoiced)-est_xv),'+')
    grid
    title(['Voiced Segment for p= ' num2str(p)])
    xlabel('Sample Number')
    ylabel('Amplitude')
    legend('Original signal','LPC estimate','error')
    
    figure(15);
    subplot(3,1,j);
    plot(1:L,bufferedAudio(:,chosenUnvoiced),1:L,est_xu,'--',1:L,abs(bufferedAudio(:,chosenUnvoiced)-est_xu),'+')
    grid
    title(['Unvoiced Segment for p= ' num2str(p)])
    xlabel('Sample Number')
    ylabel('Amplitude')
    legend('Original signal','LPC estimate','error')
    for i=1:length(a1)-1
        den2 = [a1(i) zeros(1,length(a1)-i) ];
        num2 = 1;               % denominator coefficients
        Hfilter = Hfilter + tf(den2,num2, 1/targetFs );
        %Hfilter=Hfilter+tf([  a1(i) zeros(1,length(a1)-i) ],1 ,  1/targetFs);
    end
    Hfilter = Hfilter*tf([1 zeros(1,length(a1))],[0 1],1/targetFs)^-1;
    %Hfilter=Hfilter*tf([1 zeros(1,length(a1))],[0 1],1/targetFs)^-1;
    if p==8
        Hfilter8=1/(1-Hfilter);
    elseif p==12
        Hfilter12=1/(1-Hfilter);
    elseif p==16
        Hfilter16=1/(1-Hfilter);
    end
end


figure(200)
subplot(3,1,1)
pzmap(Hfilter8)
subplot(3,1,2)
pzmap(Hfilter12)
subplot(3,1,3)
pzmap(Hfilter16)


figure(201)
subplot(3,1,1)
impulse(Hfilter8)
subplot(3,1,2)
impulse(Hfilter12)
subplot(3,1,3)
impulse(Hfilter16)



%%
%Sarting from this point there is the possibility of applying machine
%learning models 10.4
EnergySTA = zeros(1, numFrames);
shortTimeLogEnergy = zeros(1, numFrames);
zeroCrossingRate = zeros(1, numFrames);
C1 = zeros(1, numFrames);
Error_pred = zeros(1, numFrames);

% Define parameter e
epsilon = 1e-5; % Small value to prevent log(0)

for i = 1:numFrames
    currentFrame = bufferedAudio(:, i);
    
    % 1. Short-time log energy (E)
    EnergySTA(i) = (1/L)*sum(currentFrame);
    shortTimeLogEnergy(i) = 10 * log10(EnergySTA(i) + epsilon);
    
    % 2. Short-time zero crossing count (Z)
    zeroCrossings = sum(abs(diff(sign(bufferedAudio(:, i)))) > 0);
    zeroCrossingRate(i) = zeroCrossings / (L / 100); % Convert to per 10 msec interval
    
    % 3. Normalized short-time autocorrelation coefficient at unit sample delay (C1)
    C1_denominator = 0;
    C1_numerator = 0;
    for j=1:L-1
        C1_numerator = C1_numerator + sum(bufferedAudio(j,i)* bufferedAudio(j+1, i));
        C1_denominator =C1_denominator + sqrt(sum(bufferedAudio(j,i).^2) * sum(bufferedAudio(j+1, i).^2)); 
    end
    C1(i) = C1_numerator / C1_denominator;
    
    % 4. First predictor coefficient, alpha1, of a 12-pole LPC analysis
    p = 12; % LPC order
    r = xcorr(bufferedAudio(:,i), p);
    R = r(p+1:2*p+1);
    A = toeplitz(R(1:end-1));
    b = -R(2:end);
    alpha = linsolve(A, b);

    % 5. Normalized LPC log prediction error (E(p))
    Rk=0;
    for k=1:p
       
        for j=1:L-k
            Rk = Rk + bufferedAudio(j,i)* bufferedAudio(j+k, i);
        end
        Rk=alpha(k)*Rk/L;
    end
    Error_pred(i)= 10 * log10(epsilon + R(1) - Rk);
  
end


% Plot the original, resampled, and filtered waveforms
timeFiltered = (0:length(filteredAudio)-1) / targetFs;
figure(4);
plot(timeOriginal, audioData);
hold on;
plot(timeResampled, resampledAudio, '--r');
hold off;
title('Original and Resampled Audio');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Original Audio','Resampled Audio')


figure(5);
subplot(3,1,1);
plot(timeOriginal, audioData);
title('Original Audio');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3,1,2);
plot(timeResampled, resampledAudio);
title('Resampled Audio');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(3,1,3);
plot(timeFiltered, filteredAudio);
title('Highpass Filtered Audio');
xlabel('Time (s)');
ylabel('Amplitude');

figure(6);
plot(timeResampled, resampledAudio);
hold on;
plot(timeResampled, filteredAudio, '--r');
hold off;
title('Resampled and Filtered Audio');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Resampled Audio','Filtered Audio')

figure(7);
subplot(2,1,1);
plot(1:numFrames, Energy);
title('Short-Time Energy with Hamming Window');
xlabel('Time (s)');
ylabel('Energy');

subplot(2,1,2);
plot(1:numFrames, zeroCrossingRate);
title('Short-Time Zero Crossing Rate with Hamming Window');
xlabel('Time (s)');
ylabel('Zero Crossing Rate Zc');


figure(9);
subplot(4,1,1);
plot(1:numFrames, shortTimeLogEnergy);
title('Short-Time Log Energy');
xlabel('Frame Number');
ylabel('Log Energy');

subplot(4,1,2);
plot(1:numFrames, zeroCrossingRate);
title('Zero Crossing Rate');
xlabel('Frame Number');
ylabel('Zc');


subplot(4,1,3);
plot(1:numFrames, C1);
title(' Normalized short-time autocorrelation coefficient at unit sample delay');
xlabel('Frame Number');
ylabel('C1');


subplot(4,1,4);
plot(1:numFrames, Error_pred);
title(' Error Prediction ');
xlabel('Frame Number');
ylabel('Error');




figure(13);
stem(lags(:,200)/fs,autocorrValues(:,200))
title('Stem-and-Leaf Plot of Autocorrelation Function');
xlabel('Lag');
ylabel('Autocorrelation Value');
