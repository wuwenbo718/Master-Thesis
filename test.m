clc,clear;

%%

path = './data/P812_M050_2_B_FoG_trial_1_emg.csv';
M = csvread(path,1,0);
Fs=1e3;
f1 = 50;
f3 = 100;
rp = 0.1;
rs = 30;
wp=2*pi*f1/Fs;
ws=2*pi*f3/Fs;
[n,wn]=cheb1ord(wp/pi,ws/pi,rp,rs);
[bz1,az1]=cheby1(n,rp,wp/pi);
fn=1000;
ap=0.1;
as=60;
wp=50;
ws=200; %输入滤波器条件
wpp=wp/(fn/2);
wss=ws/(fn/2);  %归一化;
[nwn]=buttord(wpp,wss,ap,as); %计算阶数截止频率
[b,a]=butter(n,wn);

Mf = filter(b,a,M(:,4));
%[s,f] = stft(sig);
%stft(sig,1e6,'Window',hamming(100,'periodic'),'OverlapLength',80,'FFTLength',100);
figure();
subplot(2,1,1);
plot(M(:,1),M(:,4));
subplot(2,1,2);
plot(M(:,1),Mf);

%%
figure();
subplot(2,2,1);
plot(M(1:4401,1),M(1:4401,4));
subplot(2,2,2);
plot(M(4402:5122,1),M(4402:5122,4));
subplot(2,2,3);
plot(M(5123:5161,1),M(5123:5161,4));
subplot(2,2,4);
plot(M(5162:7842,1),M(5162:7842,4));

figure();
subplot(2,2,1);
plot(M(1:4401,1),Mf(1:4401));
subplot(2,2,2);
plot(M(4402:5122,1),Mf(4402:5122));
subplot(2,2,3);
plot(M(5123:5161,1),Mf(5123:5161));
subplot(2,2,4);
plot(M(5162:7842,1),Mf(5162:7842));

%%

%subplot(2,2,1);
figure();
cwt(M(1:4401,4),1e3);
%subplot(2,2,2);
figure();
cwt(M(4402:5122,4),1e3);
%subplot(2,2,3);
figure();
cwt(M(5123:5161,4),1e3);
%subplot(2,2,4);
figure();
cwt(M(5162:7842,4),1e3);

%%
figure();
cwt(Mf(1:4401),1e3);
figure();
cwt(Mf(4402:5122),1e3);
figure();
cwt(Mf(5123:5161),1e3);
figure();
cwt(Mf(5162:7842),1e3);

%%
figure();
[c,l] = wavedec(M(4402:5122,4),3,'db2');
approx = appcoef(c,l,'db2');
[cd1,cd2,cd3] = detcoef(c,l,[1 2 3]);
subplot(4,1,1)
plot(approx)
title('Approximation Coefficients')
subplot(4,1,2)
plot(cd3)
title('Level 3 Detail Coefficients')
subplot(4,1,3)
plot(cd2)
title('Level 2 Detail Coefficients')
subplot(4,1,4)
plot(cd1)
title('Level 1 Detail Coefficients')
figure();
plot(M(4402:5122,1),M(4402:5122,4));

%%
sig.time = M(:,1);
sig.signals.values = M(:,4);
sig.signals.dimensions = 1;