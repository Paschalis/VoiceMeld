clear all;clc;close all;
% Part 1
T = 0.008;                      % Sampling period
Fs = 10^4;                      % Sampling rate
Np = 80;                        % Delay or spacing of Np samples
Th = 3;                          % Horizon in sec
%Again=5000;
%Again=1;                       % for octave
Again=0.0001;            % for Matlab

%ksize=Np;
ksize=10000*Th/Np;
k=linspace(0,ksize-1,ksize);
nsize = Np*ksize;
n = linspace(1,nsize,nsize);    % Create a discrete time vector
b=0;
pn = zeros(1,nsize);

for nn=n
    for kk=k
        % Approximate Dirac delta
        if nn==kk*Np
            pn(nn)=0.9999^kk;
        end


    end
end

% Plot the series using a stem plot
figure(1)
subplot(3,1,1)
stem(n/Fs, pn);
title('one-sided quasi-periodic impulse train - Voiced Excitation');
xlabel('time nT [sample*period] in s');
ylabel('p[n] Series');
xlim([0 0.05])


% Define the coefficients of the system
num1 = [1 ,zeros(1,80)] ;                       % numerator coefficients
den1 = [1, zeros(1,79), -0.9999];               % denominator coefficients

% Create the transfer function
sys1= tf(num1, den1, T);        % 'T' indicates the sample time (1 for discrete-time)

num2 = [1,0 ];                                  % numerator coefficients
den2 = [1, zeros(1,79), -0.9999];               % denominator coefficients
sys2 = tf(num2,den2,1/Fs,'Variable','z^-1');



Yp=fft(pn,length(pn));                          % DFT on input signal
freq = (0:length(pn)-1)*(Fs/length(pn));        % Frequency domain

% Amplitude P(e^(j?))
Pp = abs(Yp);

figure(1);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freq, Pp);
title('Voiced excitation spectrum - Magnitude of Frequency Response w/ Fundamental Frequency F_{0} = 125 Hz');
xlabel('Frequency (Hz)');
ylabel('|P(e^{j\omega})|');
xlim([0 5000]);
grid on;


Pz=sys2;
% Get the poles and zeros
[polesysp, zerosysp] = pzmap(Pz);
figure(1)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysp), imag(zerosysp), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysp), imag(polesysp), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Voiced Excitation P(z)' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');




% Part 2
% glottal pulse - Rossenberg model
N1 = 25;
N2 = 10;

% glottal pulse sections
horizonn = 6;                      % Parameter of the horizon to be included
ng = linspace(0,N1+N2-2+horizonn,N1+N2-1+horizonn);
gn = zeros(1,size(ng,2));
for i=ng
    if i<=N1-1
        gn(i+1) = 0.5*(1 - cos(pi*(i+1) / N1));
    elseif i>=N1 && i<=N1+N2-2
        gn(i+1) = cos(0.5*pi*(i-(N1-1))/N2);
    else
        gn(i+1) = 0;
    end
end
%gnn=[0.5*(1-cos(pi*(gn1+1)/gN1)), cos(0.5*pi*(gn2+1-gN1)/gN2)]

figure(2);
% Plot of glottal pulse
subplot(3,1,1)
stem(1000*ng/Fs, gn);                                           %10^-4 because Fs and 10^3 because ms
title('Glottal Pulse');
xlabel('time nT [sample*period] in ms');
ylabel('Amplitude');


Yg=fft(gn,length(gn));                                     % DFT on input signal
freq = (0:length(gn)-1)*(Fs/length(gn));        % Frequency domain

% Amplitude P(e^(j?))
Pg = abs(Yg);

figure(2);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freq, log(Pg));
title('Glottal pulse spectrum');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|G(e^{j\omega})|)');
xlim([0 5000]);
grid on;


syms z;
% Create a symbolic polynomial using the coefficients
%{
    z transform:
    F(z) = \sum_{n=0}^{\infty} \frac{f(n)}{z^n}
%}

Gzz=0;
Gz=0;
for i=n+1
    if i<=N1
        %Gzz=Gzz+gn(i)*z^(-(i-1));
    elseif i>=N1+1 && i<=N1+N2-2
        %Gzz=Gzz+gn(i)*z^(-(i-1));
    else
        %Gzz=Gzz+gn(i)*z^(-(i-1));
    end
end
N=n(1:34);
% The goal of this part of the algo is to create a sequence of
% some tfs that are interpreted in the z domain
% They are gn(1)*z^38+gn(2)*z^37+...
for i=N+1
    Gz=Gz+tf([  gn(i) zeros(1,N(end)-i) ],1 ,  1/Fs);
end
% Opposite direction sequence entails that we need to
% multiply   by some tf z^-39
Gz=Gz*tf([1 zeros(1,N(end))],[0 1],1/Fs)^-1;

%  Plot the poles and zeros of the Z-transform
%  Compute the zeros and poles
%  Plot zeros and poles
%  Convert the symbolic expression to a transfer function
% [num,den] = numden(Gz); % Get the numerator and denominator
% num = sym2poly(num); % Convert symbolic numerator to polynomial
% % Solve the equation num = 0
% roots = vpasolve(num == 0, z);
% den = sym2poly(den); % Convert symbolic denominator to polynomial
% Gz_tf = tf(num,den); % Create the transfer function

% Get the poles and zeros
[polesysg, zerosysg] = pzmap(Gz);
figure(2)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysg), imag(zerosysg), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysg), imag(polesysg), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Glottal Pulse Gz' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');


%figure(22)
%zplane(gn(1:34),1)
%legend('zeros','poles')
%title('Zeros and Poles Plot - Glottal Pulse Gz' );
%xlabel('Real Part');
%ylabel('Imaginary Part');




%B3
% Part 3
% glottal pulse - Rossenberg model

% 1 /AO/
Fk=[570,840,2410];
sigmak=[60,100,120]./2;

% 2 /IY/
%Fk=[270,2290,3010];
%sigmak=[60,100,120]./2;

% 3 /UH/
%Fk=[440,1020,2240];
%sigmak=[60,100,120]./2;

% 4 /EH/
%Fk=[530,1840,2480];
%sigmak=[60,100,120]./2;

% 5 /AH/
%Fk=[520,1190,2390];
%sigmak=[60,100,120]./2;

% 6 /IH/
%Fk=[390,1990,2550];
%sigmak=[60,100,120]./2;

% Example /AE/
%Fk=[660,1720,2410,3500,4500];
%sigmak=[60,100,120,175,250]./2;
Vzz=1;
Vz_ao=1;
for i=1:length(Fk)
    %Vzz=Vzz*(1-2*exp(-2*pi*sigmak(i)*(1/Fs))*cos(2*pi*Fk(i)*(1/Fs))*z^-1 + exp(-4*pi*sigmak(i)*(1/Fs))*z^-2);
    Vz_ao=Vz_ao*tf([1 0 0],[1, -2*exp(-2*pi*sigmak(i)*(1/Fs))*cos(2*pi*Fk(i)*(1/Fs)),exp(-4*pi*sigmak(i)*(1/Fs))],1/Fs);
end


% Get the poles and zeros
[polesysao, zerosysao] = pzmap(Vz_ao);
figure(3)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysao), imag(zerosysao), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysao), imag(polesysao), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Vocal Tract Impulse Response V(z) /AO/');
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');



figure(3)
subplot(3,1,1)
% Find the impulse response
[vnao, nvnao] = impulse(Vz_ao);
%[vn, nvn] = impulse(Vz,[ng(1)*1000/Fs ng(end)*1000/Fs]);

% Plot the impulse response
stem(nvnao*1000, vnao);                          %1/Fs because T and 1000 because ms
title('Vocal Tract Impulse Response v[n] /AO/');
ylabel('Amplitude v[n] series ');
xlabel('time nT [sample*period] in ms');


% Extract numerator and denominator coefficients from the transfer function Vz
%[numerator, denominator] = tfdata(Vz, 'v');
% Compute the impulse response using impz
%[history, time] = impz(numerator, denominator);

% Plot the impulse response
%figure(3)
%subplot(3,1,1)
%stem(time, history);
%title('Impulse Response');
%xlabel('Time');
%ylabel('Amplitude');
%xlim([0 300])



Yvao=fft(vnao,length(vnao));                                       % DFT on input signal
freqvao = (0:length(vnao)-1)*(Fs/length(vnao));        % Frequency domain

% Amplitude P(e^(j?))
Nvao = abs(Yvao);

figure(3);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqvao, log(Nvao));
title('Vocal Tract Frequency Response /AO/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|V(e^{j\omega})|)');
xlim([0 5000]);
grid on;







% B4

Rz=tf( [1 -0.96] , [1 0] ,1/Fs);
nr=ng;
rn=zeros(1,length(nr));
for i = nr
   if i == 1
       rn(i+1) = -0.96;
   else
       rn(i+1)=0;
   end
end



% Plot the series using a stem plot
figure(4)
subplot(3,1,1)
stem(1000*nr/Fs, rn);
title('Radiation Load');
xlabel('time nT [sample*period] in ms');
ylabel('r[n] Series');


[rrn,nnr]=impulse(Rz,40);
%Yr=fft(rn(2:end),length(rn)-1);                          % DFT on input signal
%freq = (0:length(rn)-2)*(Fs/length(rn));        % Frequency domain
Yr=fft(rrn,length(rrn));
freqr = (0:length(rrn)-1)*(Fs/length(rrn));

% Amplitude P(e^(j?))
Pr = abs(Yr);

figure(4);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqr, log(Pr));
title('Radiation Load Frequency Response');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|R(e^{j\omega})|)');
xlim([0 5000]);
grid on;



% Get the poles and zeros
[polesysr, zerosysr] = pzmap(Rz);
figure(4)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysr), imag(zerosysr), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysr), imag(polesysr), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Radiation Load R(z)' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');




% B5
%Again is defined in the beginning of the script

Sz_ao=Gz*Vz_ao*Rz*Again*Pz;
Hz_ao=Gz*Vz_ao*Rz*Again;
%[ssn,ttn]=impulse(Sz,0.039);
%[hn,~]=impulse(Hz,0.039);
[ssnao,ttnao]=impulse(Sz_ao);
[hnao,~]=impulse(Hz_ao);

figure(5)
subplot(3,1,1)
plot(ttnao*1000, ssnao);
title('Output of speech model system s[n] /AO/');
xlabel('time nT [sample*period] in ms');
ylabel('s[n] Series');
xlim([0 40])



%Yr=fft(rn(2:end),length(rn)-1);                          % DFT on input signal
%freq = (0:length(rn)-2)*(Fs/length(rn));        % Frequency domain
%Ys=fft(ssn(76:end),length(ssn(76:end)));
%freq = (0:length(ssn(76:end))-1)*(Fs/length(ssn(76:end)));
Ysao=fft(ssnao,length(ssnao));
Yhao=fft(hnao,length(hnao));
freqsao = (0:length(ssnao)-1)*(Fs/length(ssnao));
freqhao = (0:length(hnao)-1)*(Fs/length(hnao));


% Amplitude P(e^(j?))
Psao = abs(Ysao);
Phao = abs(Yhao);

figure(5);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqsao, log(Psao));
hold on;
plot(freqhao, log(Phao),'--r');
hold off;
title('Output of speech model system /AO/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|S(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Get the poles and zeros
[polesyssao, zerosyssao] = pzmap(Sz_ao);
figure(5)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosyssao), imag(zerosyssao), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesyssao), imag(polesyssao), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Output of speech model system S(z) /AO/' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');







% B6

%sound(ssnah)
%sound(ssneh)
%sound(ssnih)
%sound(ssniy)
%sound(ssnuh)
%sound(ssnao)




% B7
% IY

% glottal pulse - Rossenberg model

% 2 /IY/
Fk=[270,2290,3010];
sigmak=[60,100,120]./2;

Vz_iy=1;
for i=1:length(Fk)
    Vz_iy=Vz_iy*tf([1 0 0],[1, -2*exp(-2*pi*sigmak(i)*(1/Fs))*cos(2*pi*Fk(i)*(1/Fs)),exp(-4*pi*sigmak(i)*(1/Fs))],1/Fs);
end


% Get the poles and zeros
[polesysiy, zerosysiy] = pzmap(Vz_iy);
figure(6)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysiy), imag(zerosysiy), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysiy), imag(polesysiy), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Vocal Tract Impulse Response V(z) /IY/');
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');



figure(6)
subplot(3,1,1)
% Find the impulse response
[vniy, nvniy] = impulse(Vz_iy);
%[vn, nvn] = impulse(Vz,[ng(1)*1000/Fs ng(end)*1000/Fs]);

% Plot the impulse response
stem(nvniy*1000, vniy);                          %1/Fs because T and 1000 because ms
title('Vocal Tract Impulse Response v[n] /IY/');
ylabel('Amplitude v[n] series ');
xlabel('time nT [sample*period] in ms');


Yviy=fft(vniy,length(vniy));                                       % DFT on input signal
freqviy = (0:length(vniy)-1)*(Fs/length(vniy));        % Frequency domain

% Amplitude P(e^(j?))
Nviy = abs(Yviy);

figure(6);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqviy, log(Nviy));
title('Vocal Tract Frequency Response /IY/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|V(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Sz for IY

Sz_iy=Gz*Vz_iy*Rz*Again*Pz;
Hz_iy=Gz*Vz_iy*Rz*Again;
[ssniy,ttniy]=impulse(Sz_iy);
[hniy,~]=impulse(Hz_iy);

figure(7)
subplot(3,1,1)
plot(ttniy*1000, ssniy);
title('Output of speech model system s[n] /IY/');
xlabel('time nT [sample*period] in ms');
ylabel('s[n] Series');
xlim([0 40])


Ysiy=fft(ssniy,length(ssniy));
Yhiy=fft(hniy,length(hniy));
freqsiy = (0:length(ssniy)-1)*(Fs/length(ssniy));
freqhiy = (0:length(hniy)-1)*(Fs/length(hniy));


% Amplitude P(e^(j?))
Psiy = abs(Ysiy);
Phiy = abs(Yhiy);

figure(7);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqsiy, log(Psiy));
hold on;
plot(freqhiy, log(Phiy),'--r');
hold off;
title('Output of speech model system /IY/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|S(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Get the poles and zeros
[polesyssiy, zerosyssiy] = pzmap(Sz_iy);
figure(7)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosyssiy), imag(zerosyssiy), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesyssiy), imag(polesyssiy), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Output of speech model system S(z) /IY/' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');






% UH

% glottal pulse - Rossenberg model

% 3 /UH/
Fk=[440,1020,2240];
sigmak=[60,100,120]./2;

Vz_uh=1;
for i=1:length(Fk)
    Vz_uh=Vz_uh*tf([1 0 0],[1, -2*exp(-2*pi*sigmak(i)*(1/Fs))*cos(2*pi*Fk(i)*(1/Fs)),exp(-4*pi*sigmak(i)*(1/Fs))],1/Fs);
end


% Get the poles and zeros
[polesysuh, zerosysuh] = pzmap(Vz_uh);
figure(8)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysuh), imag(zerosysuh), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysuh), imag(polesysuh), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Vocal Tract Impulse Response V(z) /UH/');
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');



figure(8)
subplot(3,1,1)
% Find the impulse response
[vnuh, nvnuh] = impulse(Vz_uh);
%[vn, nvn] = impulse(Vz,[ng(1)*1000/Fs ng(end)*1000/Fs]);

% Plot the impulse response
stem(nvnuh*1000, vnuh);                          %1/Fs because T and 1000 because ms
title('Vocal Tract Impulse Response v[n] /UH/');
ylabel('Amplitude v[n] series ');
xlabel('time nT [sample*period] in ms');


Yvuh=fft(vnuh,length(vnuh));                                       % DFT on input signal
freqvuh = (0:length(vnuh)-1)*(Fs/length(vnuh));        % Frequency domain

% Amplitude P(e^(j?))
Nvuh = abs(Yvuh);

figure(8);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqvuh, log(Nvuh));
title('Vocal Tract Frequency Response /UH/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|V(e^{j\omega})|)');
xlim([0 5000]);
grid on;

% Sz for UH

Sz_uh=Gz*Vz_uh*Rz*Again*Pz;
Hz_uh=Gz*Vz_uh*Rz*Again;
[ssnuh,ttnuh]=impulse(Sz_uh);
[hnuh,~]=impulse(Hz_uh);

figure(9)
subplot(3,1,1)
plot(ttnuh*1000, ssnuh);
title('Output of speech model system s[n] /UH/');
xlabel('time nT [sample*period] in ms');
ylabel('s[n] Series');
xlim([0 40])


Ysuh=fft(ssnuh,length(ssnuh));
Yhuh=fft(hnuh,length(hnuh));
freqsuh = (0:length(ssnuh)-1)*(Fs/length(ssnuh));
freqhuh = (0:length(hnuh)-1)*(Fs/length(hnuh));


% Amplitude P(e^(j?))
Psuh = abs(Ysuh);
Phuh = abs(Yhuh);

figure(9);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqsuh, log(Psuh));
hold on;
plot(freqhuh, log(Phuh),'--r');
hold off;
title('Output of speech model system /UH/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|S(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Get the poles and zeros
[polesyssuh, zerosyssuh] = pzmap(Sz_uh);
figure(9)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosyssuh), imag(zerosyssuh), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesyssuh), imag(polesyssuh), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Output of speech model system S(z) /UH/' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');






% EH

% glottal pulse - Rossenberg model

% 4 /EH/
Fk=[530,1840,2480];
sigmak=[60,100,120]./2;

Vz_eh=1;
for i=1:length(Fk)
    Vz_eh=Vz_eh*tf([1 0 0],[1, -2*exp(-2*pi*sigmak(i)*(1/Fs))*cos(2*pi*Fk(i)*(1/Fs)),exp(-4*pi*sigmak(i)*(1/Fs))],1/Fs);
end


% Get the poles and zeros
[polesyseh, zerosyseh] = pzmap(Vz_eh);
figure(10)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosyseh), imag(zerosyseh), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesyseh), imag(polesyseh), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Vocal Tract Impulse Response V(z) /EH/');
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');



figure(10)
subplot(3,1,1)
% Find the impulse response
[vneh, nvneh] = impulse(Vz_eh);
%[vn, nvn] = impulse(Vz,[ng(1)*1000/Fs ng(end)*1000/Fs]);

% Plot the impulse response
stem(nvneh*1000, vneh);                          %1/Fs because T and 1000 because ms
title('Vocal Tract Impulse Response v[n] /EH/');
ylabel('Amplitude v[n] series ');
xlabel('time nT [sample*period] in ms');


Yveh=fft(vneh,length(vneh));                                       % DFT on input signal
freqveh = (0:length(vneh)-1)*(Fs/length(vneh));        % Frequency domain

% Amplitude P(e^(j?))
Nveh = abs(Yveh);

figure(10);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqveh, log(Nveh));
title('Vocal Tract Frequency Response /EH/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|V(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Sz for EH

Sz_eh=Gz*Vz_eh*Rz*Again*Pz;
Hz_eh=Gz*Vz_eh*Rz*Again;
[ssneh,ttneh]=impulse(Sz_eh);
[hneh,~]=impulse(Hz_eh);

figure(11)
subplot(3,1,1)
plot(ttneh*1000, ssneh);
title('Output of speech model system s[n] /EH/');
xlabel('time nT [sample*period] in ms');
ylabel('s[n] Series');
xlim([0 40])


Yseh=fft(ssneh,length(ssneh));
Yheh=fft(hneh,length(hneh));
freqseh = (0:length(ssneh)-1)*(Fs/length(ssneh));
freqheh = (0:length(hneh)-1)*(Fs/length(hneh));


% Amplitude P(e^(j?))
Pseh = abs(Yseh);
Pheh = abs(Yheh);

figure(11);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqseh, log(Pseh));
hold on;
plot(freqheh, log(Pheh),'--r');
hold off;
title('Output of speech model system /EH/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|S(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Get the poles and zeros
[polesysseh, zerosysseh] = pzmap(Sz_eh);
figure(11)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysseh), imag(zerosysseh), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysseh), imag(polesysseh), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Output of speech model system S(z) /EH/' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');





% AH

% glottal pulse - Rossenberg model

% 5 /AH/
Fk=[520,1190,2390];
sigmak=[60,100,120]./2;


Vz_ah=1;
for i=1:length(Fk)
    Vz_ah=Vz_ah*tf([1 0 0],[1, -2*exp(-2*pi*sigmak(i)*(1/Fs))*cos(2*pi*Fk(i)*(1/Fs)),exp(-4*pi*sigmak(i)*(1/Fs))],1/Fs);
end


% Get the poles and zeros
[polesysah, zerosysah] = pzmap(Vz_ah);
figure(12)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysah), imag(zerosysah), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysah), imag(polesysah), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Vocal Tract Impulse Response V(z) /AH/');
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');



figure(12)
subplot(3,1,1)
% Find the impulse response
[vnah, nvnah] = impulse(Vz_ah);
%[vn, nvn] = impulse(Vz,[ng(1)*1000/Fs ng(end)*1000/Fs]);

% Plot the impulse response
stem(nvnah*1000, vnah);                          %1/Fs because T and 1000 because ms
title('Vocal Tract Impulse Response v[n] /AH/');
ylabel('Amplitude v[n] series ');
xlabel('time nT [sample*period] in ms');


Yvah=fft(vnah,length(vnah));                                       % DFT on input signal
freqvah = (0:length(vnah)-1)*(Fs/length(vnah));        % Frequency domain

% Amplitude P(e^(j?))
Nvah = abs(Yvah);

figure(12);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqvah, log(Nvah));
title('Vocal Tract Frequency Response /AH/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|V(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Sz for AH

Sz_ah=Gz*Vz_ah*Rz*Again*Pz;
Hz_ah=Gz*Vz_ah*Rz*Again;
[ssnah,ttnah]=impulse(Sz_ah);
[hnah,~]=impulse(Hz_ah);

figure(13)
subplot(3,1,1)
plot(ttnah*1000, ssnah);
title('Output of speech model system s[n] /AH/');
xlabel('time nT [sample*period] in ms');
ylabel('s[n] Series');
xlim([0 40])


Ysah=fft(ssnah,length(ssnah));
Yhah=fft(hnah,length(hnah));
freqsah = (0:length(ssnah)-1)*(Fs/length(ssnah));
freqhah = (0:length(hnah)-1)*(Fs/length(hnah));


% Amplitude P(e^(j?))
Psah = abs(Ysah);
Phah = abs(Yhah);

figure(13);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqsah, log(Psah));
hold on;
plot(freqhah, log(Phah),'--r');
hold off;
title('Output of speech model system /AH/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|S(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Get the poles and zeros
[polesyssah, zerosyssah] = pzmap(Sz_ah);
figure(13)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosyssah), imag(zerosyssah), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesyssah), imag(polesyssah), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Output of speech model system S(z) /AH/' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');




% IH

% glottal pulse - Rossenberg model

% 6 /IH/
Fk=[390,1990,2550];
sigmak=[60,100,120]./2;

Vz_ih=1;
for i=1:length(Fk)
    Vz_ih=Vz_ih*tf([1 0 0],[1, -2*exp(-2*pi*sigmak(i)*(1/Fs))*cos(2*pi*Fk(i)*(1/Fs)),exp(-4*pi*sigmak(i)*(1/Fs))],1/Fs);
end


% Get the poles and zeros
[polesysih, zerosysih] = pzmap(Vz_ih);
figure(14)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosysih), imag(zerosysih), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesysih), imag(polesysih), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Zeros and Poles Plot - Vocal Tract Impulse Response V(z) /IH/');
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');



figure(14)
subplot(3,1,1)
% Find the impulse response
[vnih, nvnih] = impulse(Vz_ih);
%[vn, nvn] = impulse(Vz,[ng(1)*1000/Fs ng(end)*1000/Fs]);

% Plot the impulse response
stem(nvnih*1000, vnih);                          %1/Fs because T and 1000 because ms
title('Vocal Tract Impulse Response v[n] /IH/');
ylabel('Amplitude v[n] series ');
xlabel('time nT [sample*period] in ms');


Yvih=fft(vnih,length(vnih));                                       % DFT on input signal
freqvih = (0:length(vnih)-1)*(Fs/length(vnih));        % Frequency domain

% Amplitude P(e^(j?))
Nvih = abs(Yvih);

figure(14);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqvih, log(Nvih));
title('Vocal Tract Frequency Response /IH/ ');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|V(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Sz for IH

Sz_ih=Gz*Vz_ih*Rz*Again*Pz;
Hz_ih=Gz*Vz_ih*Rz*Again;
[ssnih,ttnih]=impulse(Sz_ih);
[hnih,~]=impulse(Hz_ih);

figure(15)
subplot(3,1,1)
plot(ttnih*1000, ssnih);
title('Output of speech model system s[n] /IH/');
xlabel('time nT [sample*period] in ms');
ylabel('s[n] Series');
xlim([0 40])


Ysih=fft(ssnih,length(ssnih));
Yhih=fft(hnih,length(hnih));
freqsih = (0:length(ssnih)-1)*(Fs/length(ssnih));
freqhih = (0:length(hnih)-1)*(Fs/length(hnih));


% Amplitude P(e^(j?))
Psih = abs(Ysih);
Phih = abs(Yhih);

figure(15);
subplot(3,1,2)
% Results representation for the range 0 to 5 kHz
plot(freqsih, log(Psih));
hold on;
plot(freqhih, log(Phih),'--r');
hold off;
title('Output of speech model system /IH/');
xlabel('Frequency (Hz)');
ylabel('log_{e}(|S(e^{j\omega})|)');
xlim([0 5000]);
grid on;


% Get the poles and zeros
[polesyssih, zerosyssih] = pzmap(Sz_ih);
figure(15)
subplot(3,1,3)
% Plot zeros with a specific color, say, red 'r'
plot(real(zerosyssih), imag(zerosyssih), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
% Plot poles with a different color, say, blue 'b'
plot(real(polesyssih), imag(polesyssih), 'bx', 'MarkerSize', 10, 'LineWidth', 2);
title('Output of speech model system S(z) /IH/' );
xlabel('Real Part');
ylabel('Imaginary Part');
grid on;
theta = 0:0.01:2*pi;
x = cos(theta);
y = sin(theta);
plot(x, y, 'r:');
hold off
legend('Zeros', 'Poles');







% B8 & B9

figure(16)
svowelsappend=[ssnao(1:find(ttnao==0.5))' , ssniy(1:find(ttniy==0.5))' , ssnuh(1:find(ttnuh==0.5))' , ssneh(1:find(ttneh==0.5))' , ssnah(1:find(ttnah==0.5))' , ssnih(1:find(ttnih==0.5))'];
plot(svowelsappend)
title('Combination of /AO/ /IY/ /UH/ /EH/ /AH/ /IH/');
xlabel('time 10^{-4} sec');
ylabel('combined s[n] series');

audiowrite('output.wav', svowelsappend, Fs);
sound(svowelsappend)

% B10

% Load the recorded audio file
[audioData, fs] = audioread('output.wav');

% Parameters for the spectrogram
window10ms = hamming(round(fs * 0.01),'symmetric'); % 10 msec Hamming window
window100ms = hamming(round(fs * 0.05),'symmetric'); % 100 msec Hamming window
overlap = round(fs * 0.005); % 5 msec overlap

% Spectrogram with 10 msec Hamming window and 5 msec overlap
figure(17);
subplot(2,1,1);
spectrogram(audioData, window10ms, overlap, [], fs, 'yaxis');
title('Spectrogram with 10ms Hamming Window and 5ms Overlap');
xlabel('Time (s)');
ylabel('Frequency (kHz)');
subplot(2,1,2);
% Spectrogram with 50 msec Hamming window and 5 msec overlap
spectrogram(audioData, window100ms, overlap, [], fs, 'yaxis');
title('Spectrogram with 50ms Hamming Window and 5ms Overlap');
xlabel('Time (s)');
ylabel('Frequency (kHz)');