% Description:
%       ��ʵ����С�����׿۳��˸�˹������׿۳��������������E���Ƚϼ���С����
%   ���ܹ�������˹����ĵ���������
% Autor:
%   XingHua.He
% History:
%   2020.11.18

clc; clear; close all;
% =================================================================================================
% ========================================== ��ȡ���� =======================================
fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���');
dirOutput = dir(fullfile(fileFolder,'*.txt'));
filenames = {dirOutput.name};
energy = zeros(1,2048);
energy_x = 1:2048;
% Load all .txt files.
for i = 1:59
    energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���\' , filenames{i}));
    if i == 38
        energy(i,:) = energy_buff(:,1)';
    else
        energy(i,:) = energy_buff(:,2)';
    end
    break;
end
% =================================================================================================
% ========================================== ���ɸ�˹����� =======================================
% Conv-kernel size.
dimension = 5;
% Conv-kernel parameter.
sigma = 1;
kernel = zeros(1, dimension);
for loop = 1:dimension
    x_axis = -floor(dimension/2) + loop - 1;
    kernel(loop) = 1 / sigma * exp(-x_axis^2 / (2*sigma^2));
end
kernel = kernel ./ sum(kernel);
% =================================================================================================
% ================================== ���ø�˹������׿۳� =======================================
% Iter number.
iteration = 1000;

background_guass = energy;
for i = 1:iteration
    background_buff = conv(kernel, background_guass(1, :));
    background_buff = background_buff((floor(dimension/2)+1):(floor(dimension/2)+1+2047));
    for j = 1:2048
        if background_buff(j) < background_guass(1,j)
            background_guass(1,j) = background_buff(j);
        else
            background_guass(1,j) = background_guass(1,j);
        end
    end
end
% =================================================================================================
% ================================== ����MorletС���˾�����׿۳� ===========================
lb = -2;
ub = 2;
n = 5;
[psi,xval] = morlet(lb,ub,n);
mw_kernel = psi;
mw_kernel = mw_kernel ./ sum(mw_kernel);

background_mw = energy;
for i = 1:iteration
    background_buff_mw = conv(mw_kernel, background_mw(1, :));
    background_buff_mw = background_buff_mw((floor(dimension/2)+1):(floor(dimension/2)+1+2047));
    for j = 1:2048
        if background_buff_mw(j) < background_mw(1,j)
            background_mw(1,j) = background_buff_mw(j);
        else
            background_mw(1,j) = background_mw(1,j);
        end
    end
end
% =================================================================================================
% ================================== ����С�����׿۳� =======================================
background_wavelet = energy;

[c,l] = wavedec(background_wavelet ,7,'sym4');%��߶�һά��ɢС�����ع���1-7��ƽ�ϵ�� db5
a7 = wrcoef('a',c,l,'sym4',7);
a6 = wrcoef('a',c,l,'sym4',6);
a5 = wrcoef('a',c,l,'sym4',5);
a4 = wrcoef('a',c,l,'sym4',4);
a3 = wrcoef('a',c,l,'sym4',3);
a2 = wrcoef('a',c,l,'sym4',2);
a1 = wrcoef('a',c,l,'sym4',1);

for iloop = 1:50
    for i = 1:length(background_wavelet)
        if(background_wavelet(i) > a7(i) && a7(i) > 0)
            background_wavelet(i)=a7(i);
        end
    end
    [c,l] = wavedec(background_wavelet, 7, 'sym4');%�ع���1-7��ƽ�ϵ��
    a7 = wrcoef('a', c, l, 'sym4', 7);
end
% =================================================================================================
% ============================== �����ε���С�������˹ ====================================
background_max = energy;
% error.
error = 1e-10;

[c,l] = wavedec(background_max ,7,'sym4');%��߶�һά��ɢС�����ع���1-7��ƽ�ϵ�� db5
a7 = wrcoef('a',c,l,'sym4',7);
a6 = wrcoef('a',c,l,'sym4',6);
a5 = wrcoef('a',c,l,'sym4',5);
a4 = wrcoef('a',c,l,'sym4',4);
a3 = wrcoef('a',c,l,'sym4',3);
a2 = wrcoef('a',c,l,'sym4',2);
a1 = wrcoef('a',c,l,'sym4',1);
for i = 1:length(background_max)
    if(background_max(i) > a7(i) && a7(i) > 0)
        background_max(i)=a7(i);
    end
end

count = 0;
while(1)
    background_buff_max = conv(kernel, background_max(1, :));
    background_buff_max = background_buff_max((floor(dimension/2)+1):(floor(dimension/2)+1+2047));
    for j = 1:2048
        if background_buff_max(j) < background_max(1,j)
            background_max(1,j) = background_buff_max(j);
        else
            background_max(1,j) = background_max(1,j);
        end
    end
    
    % Computed error.
    diff = sqrt(sum((background_max - background_guass).^2) / length(background_guass));
    if diff <= error
        break;
    end
    count = count + 1; 
    if count == iteration
        break;
    end
end
count
% =================================================================================================
% ================================== ��ͼ =======================================
figure(1);
hold on;
plot(1:1:2048, energy);
plot(1:1:2048, background_guass);
% plot(1:1:2048, background_mw);

% plot(1:1:2048, background_wavelet);
% plot(1:1:2048, background_max);