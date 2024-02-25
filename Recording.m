% Set the sample rate
fs = 16000; % or 22050 for 22.05 kHz

% Set the recording duration in seconds
recordingDuration = 3;

% Create an audio recorder object
recorder = audiorecorder(fs, 16, 1); % 16 bits per sample, 1 channel (monophonic)

% Record the audio
disp('Start speaking...');
recordblocking(recorder, recordingDuration);
disp('End of recording');

% Get the recorded data
audioData = getaudiodata(recorder);

% Play the recorded audio
sound(audioData, fs);

% Save the recorded audio to a file (optional)
audiowrite('recorded_audio_1.wav', audioData, fs);