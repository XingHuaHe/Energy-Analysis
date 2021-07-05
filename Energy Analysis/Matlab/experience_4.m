% Description:
%       本实验内容
%       （1）迭代窗小波本底扣除
%       （2）改进的迭代窗小波本底扣除
%       （3）改进的迭代窗小波+高斯卷积本底扣除
% 
%       未实现：
%       （4）融合 ‘迭代双树复小波’ 和 ‘迭代窗小波’ 本底扣除
%       （5）迭代双树复小波本底扣除
%       （6）迭代双树复窗小波本底扣除
%       （7）迭代双树复窗小波本底扣除+迭代高斯卷积
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
% ==========================================================================================
% ================================== 1.迭代窗小波本底扣除 ======================================
background_wavelet_adj = energy;
% 待选小波基.
wavelet_package = {'sym4', 'bior6.8'}; % {'db5', 'db8', 'db30', 'sym4', 'bior6.8'};
% 窗大小.
windows_size = 50;

% 遍历每个样本.
for j = 1:size(background_wavelet_adj, 1)
    % 用不同的小波基wavelet_package进行本底拟合.
    background_wavelet_package = zeros(length(wavelet_package), size(background_wavelet_adj, 2));
    background_wavelet_temp1 = background_wavelet_adj(j, :);
    for p = 1:length(wavelet_package)
        background_wavelet_temp2 = background_wavelet_temp1;
        [c,l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
        a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
        
        for iloop = 1:5
            for i = 1:length(background_wavelet_temp2)
                if(background_wavelet_temp2(i) > a7(i) && a7(i) > 0)
                    background_wavelet_temp2(i) = a7(i);
                end
            end
            [c, l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
            a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
        end
        
        background_wavelet_package(p, :) = background_wavelet_temp2;
    end
    
    % 根据不同的小波基，以窗的形式，提取每个窗最佳拟合本底.
    windows_count = int32(length(background_wavelet_adj(j, :)) / windows_size);
    start_count = 1;
    end_count = windows_size;
    for w = 1:windows_count
        if(w == windows_count)
            % 最后一个窗特殊处理.
            best_bg = zeros(1, length(background_wavelet_package) - start_count + 1);
            bg = 0;
            for p = 1:length(wavelet_package)
                bg_temp = sqrt(sum(background_wavelet_package(p, start_count:end)) / windows_size);
                if(p == 1)
                    bg = bg_temp;
                    best_bg(1, :) = background_wavelet_package(p, start_count:end);
                    continue;
                end
                if(bg_temp > bg)
                    best_bg(1, :) = background_wavelet_package(p, start_count:end);
                    bg = bg_temp;
                end
            end
            % 将当前窗最好的本底赋值给最终的结果矩阵对应窗.
            background_wavelet_adj(j, start_count:end) = best_bg;
        else
            % 遍历每个小波基再目前的窗的本底.
            best_bg = zeros(1, windows_size);
            bg = 0;
            for p = 1:length(wavelet_package)
                bg_temp = sqrt(sum(background_wavelet_package(p, start_count:end_count)) / windows_size);
                if(p == 1)
                    bg = bg_temp;
                    best_bg(1, :) = background_wavelet_package(p, start_count:end_count);
                    continue;
                end
                if(bg_temp > bg)
                    best_bg(1, :) = background_wavelet_package(p, start_count:end_count);
                    bg = bg_temp;
                end
            end
            
            % 将当前窗最好的本底赋值给最终的结果矩阵对应窗.
            background_wavelet_adj(j, start_count:end_count) = best_bg;
        end
        
        start_count = start_count + windows_size;
        end_count = end_count + windows_size;
    end
end
% ==========================================================================================
% ================================== 2.改进的迭代窗小波本底扣除 ==================================
background_imp_IW_wavelet = energy;
% 待选小波基.
wavelet_package = {'sym4', 'bior6.8'}; % {'db5', 'db8', 'db30', 'sym4', 'bior6.8'};
% 窗大小.
windows_size = 50;
% 局部变化窗数量.
windows_consider = 10;

% 遍历每个样本.
for j = 1:size(background_imp_IW_wavelet, 1)
    % 保存用不同的小波基wavelet_package进行本底拟合.
    background_ipm_IW_wavelet_pg = zeros(length(wavelet_package), size(background_imp_IW_wavelet, 2));
    % 保存临时数据.
    background_wavelet_temp = background_imp_IW_wavelet(j, :);
    % 改进的窗迭代小波本底扣除.
    windows_count = int32(length(background_imp_IW_wavelet(j, :)) / windows_size);
    start_count = 1;
    end_count = windows_size;
    for w = 1:windows_count
        % 情况1：当前窗 w < 小波局部小波变化考虑windows_consider.
        if w <= windows_consider
            % 初始化局部谱图.
            energy_local = zeros(1, windows_size*(windows_consider+w));
            % 将局部谱图从原普图剥离出来.
            energy_local = background_wavelet_temp(1, 1:windows_size*(windows_consider+w));

            % 对局部谱图进行迭代小波本底扣除.
            for p = 1:length(wavelet_package)
                background_wavelet_temp2 = energy_local;
                [c,l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                % 迭代小波本底扣除.
                for iloop = 1:5
                    for i = 1:length(background_wavelet_temp2)
                        if(background_wavelet_temp2(i) > a7(i) && a7(i) > 0)
                            background_wavelet_temp2(i) = a7(i);
                        end
                    end
                    [c, l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                    a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                end

                % 保存每个窗本底（窗本底采用局部的迭代近似方法）
                background_ipm_IW_wavelet_pg(p, start_count:end_count) = background_wavelet_temp2(1, start_count:end_count);
            end
        end

        % 情况2：当前窗位于情况1、2之间.
        if w > windows_consider && w < (windows_count - windows_consider)
            % 初始化局部谱图.
            energy_local = zeros(1, windows_size*(2*windows_consider+1));
            % 将局部谱图从原普图剥离出来.
            energy_local = background_wavelet_temp(1, (w-windows_consider-1)*windows_size+1:(w+windows_consider)*windows_size);

            % 对局部谱图进行迭代小波本底扣除.
            for p = 1:length(wavelet_package)
                background_wavelet_temp2 = energy_local;
                [c,l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                % 迭代小波本底扣除.
                for iloop = 1:5
                    for i = 1:length(background_wavelet_temp2)
                        if(background_wavelet_temp2(i) > a7(i) && a7(i) > 0)
                            background_wavelet_temp2(i) = a7(i);
                        end
                    end
                    [c, l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                    a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                end

                % 保存每个窗本底（窗本底采用局部的迭代近似方法）
                background_ipm_IW_wavelet_pg(p, start_count:end_count) = background_wavelet_temp2(1, windows_size*windows_consider+1:(windows_consider+1)*windows_size);
            end
        end

        % 情况3：当前窗 w > windows_count - window_consider.
        if w >= (windows_count - windows_consider)
            % 初始化局部谱图.
            energy_local = zeros(1, windows_size*(windows_consider)+length(background_wavelet_temp(1, start_count:end)));
            % 将局部谱图从原普图剥离出来.
            energy_local = background_wavelet_temp(1, (w-windows_consider-1)*windows_size+1:end);

            % 对局部谱图进行迭代小波本底扣除.
            for p = 1:length(wavelet_package)
                background_wavelet_temp2 = energy_local;
                [c,l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                % 迭代小波本底扣除.
                for iloop = 1:5
                    for i = 1:length(background_wavelet_temp2)
                        if(background_wavelet_temp2(i) > a7(i) && a7(i) > 0)
                            background_wavelet_temp2(i) = a7(i);
                        end
                    end
                    [c, l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                    a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                end

                % 保存每个窗本底（窗本底采用局部的迭代近似方法）
                if w == windows_count
                    % 最后一个窗.
                    background_ipm_IW_wavelet_pg(p, start_count:end) = background_wavelet_temp2(1, windows_size*windows_consider+1:end);
                else
                    % 非最后一个窗.
                    background_ipm_IW_wavelet_pg(p, start_count:end_count) = background_wavelet_temp2(1, windows_size*windows_consider+1:windows_size*(windows_consider+1));
                end
                
            end
        end

        start_count = start_count + windows_size;
        end_count = end_count + windows_size;
    end

    % 挑选每个窗最佳本底（采用均值最大）
    start_count = 1;
    end_count = windows_size;
    for w = 1:windows_count
        % 最后一个窗.
        if w == windows_count
            best_bg = zeros(1, length(background_ipm_IW_wavelet_pg) - start_count + 1);
            bg = 0;
            for p = 1:size(background_ipm_IW_wavelet_pg, 1)
                % 计算均值.
                if p == 1
                    bg = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end)) / length(background_ipm_IW_wavelet_pg(p, start_count:end)));
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end);
                    continue;
                end
                bg_temp = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end)) / length(background_ipm_IW_wavelet_pg(p, start_count:end)));
                if bg_temp > bg
                    % 替换均值最大的.
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end);
                end
            end
            % 将当前窗最好的本底保存.
            background_imp_IW_wavelet(j, start_count:end) = best_bg(1, :);
        else
            best_bg = zeros(1, windows_size);
            bg = 0;
            for p = 1:size(background_ipm_IW_wavelet_pg, 1)
                % 计算均值.
                if p == 1
                    bg = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end_count)) / windows_size);
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end_count);
                    continue;
                end
                bg_temp = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end_count)) / windows_size);
                if bg_temp > bg
                    % 替换均值最大的.
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end_count);
                end
                % 将当前窗最好的本底保存.
                background_imp_IW_wavelet(j, start_count:end_count) = best_bg(1, :);
            end
        end

        start_count = start_count + windows_size;
        end_count = end_count + windows_size;
    end
end
% ==========================================================================================
% ======================== 3.改进的迭代窗小波+高斯卷积本底扣除 ==================================
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
iteration = 40;

% 改进的迭代窗小波+高斯卷积
background_IIW_wave_Gass = background_imp_IW_wavelet;
% 基于高斯卷积对改进的迭代窗小波本底扣除结果进行进一步滤波.
for z = 1:size(background_imp_IW_wavelet, 1)
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
end

% ==========================================================================================
% ================== 融合 ‘迭代双树复小波’ 和 ‘迭代窗小波’ 本底扣除  =================


% =================================================================================
% ================================== 计算R2 =======================================
% 窗迭代小波扣除本底.
energy_windows_wave = energy - background_wavelet_adj;
% 增强窗迭代小波本底扣除.
energy_imp_iter_windows_wave = energy - background_imp_IW_wavelet;
% 增强窗迭代小波+高斯卷积本底扣除.
energy_IIW_wave_gass = energy - background_IIW_wave_Gass;

id = zeros(1, 59);
for i = 1:59
    id(i) = str2double(strrep(filenames{i},'.txt',''));
end
% 根据能量反算通道值,取出对应峰面积, 定义Cr，Ni，Mn，Cu元素能量.
Cr_Ka = 5.41;
Ni_Ka = 7.47;
Mn_Ka = 5.895;
Cu_Ka = 8.04;

element_type = 2; % 2:Cr 3:Mn 4:Ni 5:Cu
% Cr
window_low = 149;                 %卡窗左界
window_high = 163;                %卡窗右界
% Mn
% window_low = 165;                 %卡窗左界
% window_high = 177;                %卡窗右界
% Cu
% window_low = 227;                 %卡窗左界
% window_high = 243;                %卡窗右界

ag_window_low = 626;
ag_window_high = 647;
% 康普顿.
% energy_ag_expand = zeros(59, (ag_window_high - ag_window_low + 1));
% 迭代窗小波去本底后谱图.
energy_wavelet_expand = zeros(59, (window_high - window_low + 1));
% 增强迭代窗小波扣除本底后谱图.
energy_imp_wavelet_expand = zeros(59, (window_high - window_low + 1));
% 增强窗迭代小波+高斯卷积扣除本底后谱图.
energy_IIW_wav_gass_expand = zeros(59, (window_high - window_low + 1));
% 原谱图.
energy_expand = zeros(59, (window_high - window_low + 1));
for i = 1:59
    for j = 1:(window_high - window_low + 1)
        % 康普顿.
        % energy_ag_expand(i,j) = energy_wavelet_gauss(i, ag_window_low+j-1);
        % 迭代窗小波去本底后谱图.
        energy_wavelet_expand(i,j) = energy_windows_wave(i, window_low+j-1);
        % 增强迭代窗小波扣除本底后谱图.
        energy_imp_wavelet_expand(i,j) = energy_imp_iter_windows_wave(i, window_low+j-1);
        % 增强窗迭代小波+高斯卷积扣除本底后谱图.
        energy_IIW_wav_gass_expand(i,j) = energy_IIW_wave_gass(i, window_low+j-1);
        % 原谱图.
        energy_expand(i,j) = energy(i, window_low+j-1 );
    end
end
% 康普顿峰面积.
% energy_Ag_area = zeros(1,59);
% 迭代窗小波去本底后谱图.
energy_wavelet_area = zeros(1, 59);
% 增强迭代窗小波扣除本底后谱图.
energy_imp_wave_wind_area = zeros(1, 59);
% 增强窗迭代小波+高斯卷积扣除本底后谱图.
energy_IIW_wav_gass_area = zeros(1, 59);
% 原谱图.
energy_area = zeros(1, 59);
for i = 1:59
    % 康普顿.
    % energy_Ag_area(i) = sum(energy_ag_expand(i,:));
    % 迭代窗小波去本底后谱图.
    energy_wavelet_area(i) = sum(energy_wavelet_expand(i,:));
    % 增强迭代窗小波扣除本底后谱图.
    energy_imp_wave_wind_area(i) = sum(energy_imp_wavelet_expand(i,:));
    % 增强窗迭代小波+高斯卷积扣除本底后谱图.
    energy_IIW_wav_gass_area(i) = sum(energy_IIW_wav_gass_expand(i,:));
    % 原谱图.
    energy_area(i) = sum(energy_expand(i,:));
end

% 迭代窗小波去本底扣除，算相关系数
data = importdata('D:\Processing\Energy Analysis\Matlab\土壤\数据.txt');
contents_IW_wave = zeros(59,2);
for i = 1:59
    contents_IW_wave(i,1) = energy_wavelet_area(i);
    for j = 1:59
        if data(j,1) == id(i)
            contents_IW_wave(i,2) = data(j, element_type);
            break
        end
    end
end
R1 = corrcoef(contents_IW_wave);
R_2_1 = (R1(1,2)^2);
fprintf("迭代窗小波本底扣除： R2=%f\n", R_2_1);

% 增强迭代窗小波扣除本底,算相关系数.
contents_IIW_wave = zeros(59,2);
for i = 1:59
    contents_IIW_wave(i,1) = energy_imp_wave_wind_area(i);
    for j = 1:59
        if data(j,1) == id(i)
            contents_IIW_wave(i,2) = data(j, element_type);
            break
        end
    end
end
R2 = corrcoef(contents_IIW_wave);
R_2_2 = (R2(1,2)^2);
fprintf("增强版迭代窗小波本底扣除： R2=%f\n", R_2_2);

% 增强窗迭代小波+高斯卷积去本底扣除，算相关系数
contents_IIW_wave_gass = zeros(59,2);
for i = 1:59
    contents_IIW_wave_gass(i,1) = energy_IIW_wav_gass_area(i);
    for j = 1:59
        if data(j,1) == id(i)
            contents_IIW_wave_gass(i,2) = data(j, element_type);
            break
        end
    end
end
R3 = corrcoef(contents_IIW_wave_gass);
R_2_3 = (R3(1,2)^2);
fprintf("增强版迭代窗小波+高斯卷积本底扣除： R2=%f\n", R_2_3);


% ===============================================================================
% ============================== 绘制本底图 =======================================
figure(1);
hold on;
plot(1:1:length(energy(1,:)), energy(1, :));
plot(1:1:length(energy(1,:)), background_wavelet_adj(1,:));
plot(1:1:length(energy(1,:)), background_imp_IW_wavelet(1,:));
plot(1:1:length(energy(1,:)), background_IIW_wave_Gass(1,:));
legend("原信号", "迭代窗小波", "增强版迭代窗小波", "增强版迭代窗小波+高斯卷积");