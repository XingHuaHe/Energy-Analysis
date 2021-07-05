% Description:
%       本实验内容
%       （1）迭代小波本底扣除
%       （2）动量+迭代小波本底扣除
%       （3）双树复小波本底扣除
%       （4）双树复小波本底扣除 + 高斯
% Autor:
%   XingHua.He
% History:
%   2020.11.26

clc; clear; close all;
% ==========================================================================================
% ========================================== 读取数据 =======================================
% fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\土壤\土壤谱图\新建文件夹');
% dirOutput = dir(fullfile(fileFolder,'*.txt'));
% filenames = {dirOutput.name};
% energy = zeros(59,2048);
% energy_x = 1:2048;
% % Load all .txt files.
% for i = 1:59
%     energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\土壤\土壤谱图\新建文件夹\', filenames{i}));
%     if i == 38
%         energy(i,:) = energy_buff(:,1)';
%     else
%         energy(i,:) = energy_buff(:,2)';
%     end
% end
load('soil.mat');
energy = spectrums(:,1:2048);
contents = spectrums(:,2049:end);
% =================================================================================================
% ======================================== 迭代小波本底扣除 =====================================
% energy = energy(:,1:1300);
background_wavelet = energy;
for j = 1:size(background_wavelet, 1)
    [c,l] = wavedec(background_wavelet(j, :) ,7,'sym4');%多尺度一维离散小波，重构第1-7层逼近系数 db5
    a7 = wrcoef('a',c,l,'sym4',7);

    for iloop = 1:50
        for i = 1:length(background_wavelet(j, :))
            if(background_wavelet(j, i) > a7(i) && a7(i) > 0)
                background_wavelet(j, i) = a7(i);
            end
        end
        [c,l] = wavedec(background_wavelet(j, :), 7, 'sym4');%重构第1-7层逼近系数
        a7 = wrcoef('a', c, l, 'sym4', 7);
    end
end
% =================================================================================================
% ======================================== 动量+迭代小波本底扣除 =====================================
% energy = energy(:,1:1300);
background_m_wavelet = energy;
m = 0.5;
for j = 1:size(background_m_wavelet, 1)
    [c,l] = wavedec(background_m_wavelet(j, :) ,7,'sym4');%多尺度一维离散小波，重构第1-7层逼近系数 db5
    a7 = wrcoef('a',c,l,'sym4',7);

    for iloop = 1:50
        for i = 1:length(background_m_wavelet(j, :))
            if(background_m_wavelet(j, i) > a7(i) && a7(i) > 0)
                background_m_wavelet(j, i)=m*a7(i) + (1-m)* background_m_wavelet(j, i);
            end
        end
        [c,l] = wavedec(background_m_wavelet(j, :), 7, 'sym4');%重构第1-7层逼近系数
        a7 = wrcoef('a', c, l, 'sym4', 7);
    end
end
% ==========================================================================================
% =================================== 双树复小波本底扣除 ====================================
background_dddtree_wave = energy;
m = 0.4;
% figure(1);
% hold on;
for j = 1:size(background_dddtree_wave, 1)
    wt = dddtree('cplxdt', energy(1, :), 4, 'FSfarras', 'qshift06');
    wt.cfs{1} = zeros(1, 1024, 2);
    wt.cfs{2} = zeros(1, 512, 2);
    wt.cfs{3} = zeros(1, 256, 2);
    wt.cfs{4} = zeros(1, 128, 2);
    wt.cfs{4}(1,:,1) = zeros(1, 128);
%     wt.cfs{5} = zeros(1, 64, 2);
%     wt.cfs{6} = zeros(1, 32, 2);
%     wt.cfs{7} = zeros(1, 16, 2);
    xrec = idddtree(wt);
    
    for iloop = 1:20
        for i = 1:length(background_dddtree_wave(j, :))
            if(background_dddtree_wave(j, i) > xrec(i) && xrec(i) > 0)
                background_dddtree_wave(j, i) = m*xrec(i) + (1-m)*background_dddtree_wave(j, i);
            end
        end
%         plot(1:1:2048, background_dddtree_wave(1,:));
        
        wt = dddtree('cplxdt', background_dddtree_wave(1, :), 4, 'FSfarras', 'qshift06');
        wt.cfs{1} = zeros(1, 1024, 2);
        wt.cfs{2} = zeros(1, 512, 2);
        wt.cfs{3} = zeros(1, 256, 2);
        wt.cfs{4} = zeros(1, 128, 2);
        wt.cfs{4}(1,:,1) = zeros(1, 128);
%         wt.cfs{5} = zeros(1, 64, 2);
%         wt.cfs{6} = zeros(1, 32, 2);
%         wt.cfs{7} = zeros(1, 16, 2);
        xrec = idddtree(wt);
    end
    break;
end

% ==========================================================================================
% =============================== 双树复小波+高斯本底扣除 ====================================
% Conv-kernel size.
dimension = 5;
% Conv-kernel parameter.
sigma = 0.7;
kernel = zeros(1, dimension);
for loop = 1:dimension
    x_axis = -floor(dimension/2) + loop - 1;
    kernel(loop) = 1 / sigma * exp(-x_axis^2 / (2*sigma^2));
end
kernel = kernel ./ sum(kernel);

% Iter number.
iteration = 500;

% 改进的迭代窗小波+高斯卷积
background_IIW_wave_Gass = background_dddtree_wave;
% 基于高斯卷积对改进的迭代窗小波本底扣除结果进行进一步滤波.
for z = 1:size(background_dddtree_wave, 1)
    guass = background_IIW_wave_Gass(z, :);
    for i = 1:iteration
        background_buff = conv(kernel, guass(1, :));
        background_buff = background_buff((floor(dimension/2)+1):(floor(dimension/2)+1+length(background_buff)-dimension));
        for j = 1:length(background_buff)
            if background_buff(j) < guass(1,j)
                guass(1,j) = background_buff(j);
            else
                guass(1,j) = guass(1,j);
            end
        end
    end
    background_IIW_wave_Gass(z, :) = guass(1, :);
    break;
end




figure(1);
hold on;
plot(1:1:2048, energy(1,:));
plot(1:1:2048, background_wavelet(1,:));
plot(1:1:2048, background_m_wavelet(1,:));
plot(1:1:2048, background_dddtree_wave(1,:));
plot(1:1:2048, background_IIW_wave_Gass(1,:));
% plot(1:1:2048, energy(1,:) - background_dddtree_wave(1,:));
legend('原信号', '迭代小波本底','动量+迭代小波本底', '迭代双树复小波本底','迭代双树复小波本底+高斯');

