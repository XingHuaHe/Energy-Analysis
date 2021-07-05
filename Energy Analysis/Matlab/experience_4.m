% Description:
%       ��ʵ������
%       ��1��������С�����׿۳�
%       ��2���Ľ��ĵ�����С�����׿۳�
%       ��3���Ľ��ĵ�����С��+��˹������׿۳�
% 
%       δʵ�֣�
%       ��4���ں� ������˫����С���� �� ��������С���� ���׿۳�
%       ��5������˫����С�����׿۳�
%       ��6������˫������С�����׿۳�
%       ��7������˫������С�����׿۳�+������˹���
% Autor:
%   XingHua.He
% History:
%   2020.11.26

clc; clear; close all;
% ==========================================================================================
% ========================================== ��ȡ���� =======================================
% fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���');
% dirOutput = dir(fullfile(fileFolder,'*.txt'));
% filenames = {dirOutput.name};
% energy = zeros(59,2048);
% energy_x = 1:2048;
% % Load all .txt files.
% for i = 1:59
%     energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���\', filenames{i}));
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
% ================================== 1.������С�����׿۳� ======================================
background_wavelet_adj = energy;
% ��ѡС����.
wavelet_package = {'sym4', 'bior6.8'}; % {'db5', 'db8', 'db30', 'sym4', 'bior6.8'};
% ����С.
windows_size = 50;

% ����ÿ������.
for j = 1:size(background_wavelet_adj, 1)
    % �ò�ͬ��С����wavelet_package���б������.
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
    
    % ���ݲ�ͬ��С�������Դ�����ʽ����ȡÿ���������ϱ���.
    windows_count = int32(length(background_wavelet_adj(j, :)) / windows_size);
    start_count = 1;
    end_count = windows_size;
    for w = 1:windows_count
        if(w == windows_count)
            % ���һ�������⴦��.
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
            % ����ǰ����õı��׸�ֵ�����յĽ�������Ӧ��.
            background_wavelet_adj(j, start_count:end) = best_bg;
        else
            % ����ÿ��С������Ŀǰ�Ĵ��ı���.
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
            
            % ����ǰ����õı��׸�ֵ�����յĽ�������Ӧ��.
            background_wavelet_adj(j, start_count:end_count) = best_bg;
        end
        
        start_count = start_count + windows_size;
        end_count = end_count + windows_size;
    end
end
% ==========================================================================================
% ================================== 2.�Ľ��ĵ�����С�����׿۳� ==================================
background_imp_IW_wavelet = energy;
% ��ѡС����.
wavelet_package = {'sym4', 'bior6.8'}; % {'db5', 'db8', 'db30', 'sym4', 'bior6.8'};
% ����С.
windows_size = 50;
% �ֲ��仯������.
windows_consider = 10;

% ����ÿ������.
for j = 1:size(background_imp_IW_wavelet, 1)
    % �����ò�ͬ��С����wavelet_package���б������.
    background_ipm_IW_wavelet_pg = zeros(length(wavelet_package), size(background_imp_IW_wavelet, 2));
    % ������ʱ����.
    background_wavelet_temp = background_imp_IW_wavelet(j, :);
    % �Ľ��Ĵ�����С�����׿۳�.
    windows_count = int32(length(background_imp_IW_wavelet(j, :)) / windows_size);
    start_count = 1;
    end_count = windows_size;
    for w = 1:windows_count
        % ���1����ǰ�� w < С���ֲ�С���仯����windows_consider.
        if w <= windows_consider
            % ��ʼ���ֲ���ͼ.
            energy_local = zeros(1, windows_size*(windows_consider+w));
            % ���ֲ���ͼ��ԭ��ͼ�������.
            energy_local = background_wavelet_temp(1, 1:windows_size*(windows_consider+w));

            % �Ծֲ���ͼ���е���С�����׿۳�.
            for p = 1:length(wavelet_package)
                background_wavelet_temp2 = energy_local;
                [c,l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                % ����С�����׿۳�.
                for iloop = 1:5
                    for i = 1:length(background_wavelet_temp2)
                        if(background_wavelet_temp2(i) > a7(i) && a7(i) > 0)
                            background_wavelet_temp2(i) = a7(i);
                        end
                    end
                    [c, l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                    a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                end

                % ����ÿ�������ף������ײ��þֲ��ĵ������Ʒ�����
                background_ipm_IW_wavelet_pg(p, start_count:end_count) = background_wavelet_temp2(1, start_count:end_count);
            end
        end

        % ���2����ǰ��λ�����1��2֮��.
        if w > windows_consider && w < (windows_count - windows_consider)
            % ��ʼ���ֲ���ͼ.
            energy_local = zeros(1, windows_size*(2*windows_consider+1));
            % ���ֲ���ͼ��ԭ��ͼ�������.
            energy_local = background_wavelet_temp(1, (w-windows_consider-1)*windows_size+1:(w+windows_consider)*windows_size);

            % �Ծֲ���ͼ���е���С�����׿۳�.
            for p = 1:length(wavelet_package)
                background_wavelet_temp2 = energy_local;
                [c,l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                % ����С�����׿۳�.
                for iloop = 1:5
                    for i = 1:length(background_wavelet_temp2)
                        if(background_wavelet_temp2(i) > a7(i) && a7(i) > 0)
                            background_wavelet_temp2(i) = a7(i);
                        end
                    end
                    [c, l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                    a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                end

                % ����ÿ�������ף������ײ��þֲ��ĵ������Ʒ�����
                background_ipm_IW_wavelet_pg(p, start_count:end_count) = background_wavelet_temp2(1, windows_size*windows_consider+1:(windows_consider+1)*windows_size);
            end
        end

        % ���3����ǰ�� w > windows_count - window_consider.
        if w >= (windows_count - windows_consider)
            % ��ʼ���ֲ���ͼ.
            energy_local = zeros(1, windows_size*(windows_consider)+length(background_wavelet_temp(1, start_count:end)));
            % ���ֲ���ͼ��ԭ��ͼ�������.
            energy_local = background_wavelet_temp(1, (w-windows_consider-1)*windows_size+1:end);

            % �Ծֲ���ͼ���е���С�����׿۳�.
            for p = 1:length(wavelet_package)
                background_wavelet_temp2 = energy_local;
                [c,l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                % ����С�����׿۳�.
                for iloop = 1:5
                    for i = 1:length(background_wavelet_temp2)
                        if(background_wavelet_temp2(i) > a7(i) && a7(i) > 0)
                            background_wavelet_temp2(i) = a7(i);
                        end
                    end
                    [c, l] = wavedec(background_wavelet_temp2, 7, char(wavelet_package(p)));
                    a7 = wrcoef('a', c, l, char(wavelet_package(p)), 7);
                end

                % ����ÿ�������ף������ײ��þֲ��ĵ������Ʒ�����
                if w == windows_count
                    % ���һ����.
                    background_ipm_IW_wavelet_pg(p, start_count:end) = background_wavelet_temp2(1, windows_size*windows_consider+1:end);
                else
                    % �����һ����.
                    background_ipm_IW_wavelet_pg(p, start_count:end_count) = background_wavelet_temp2(1, windows_size*windows_consider+1:windows_size*(windows_consider+1));
                end
                
            end
        end

        start_count = start_count + windows_size;
        end_count = end_count + windows_size;
    end

    % ��ѡÿ������ѱ��ף����þ�ֵ���
    start_count = 1;
    end_count = windows_size;
    for w = 1:windows_count
        % ���һ����.
        if w == windows_count
            best_bg = zeros(1, length(background_ipm_IW_wavelet_pg) - start_count + 1);
            bg = 0;
            for p = 1:size(background_ipm_IW_wavelet_pg, 1)
                % �����ֵ.
                if p == 1
                    bg = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end)) / length(background_ipm_IW_wavelet_pg(p, start_count:end)));
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end);
                    continue;
                end
                bg_temp = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end)) / length(background_ipm_IW_wavelet_pg(p, start_count:end)));
                if bg_temp > bg
                    % �滻��ֵ����.
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end);
                end
            end
            % ����ǰ����õı��ױ���.
            background_imp_IW_wavelet(j, start_count:end) = best_bg(1, :);
        else
            best_bg = zeros(1, windows_size);
            bg = 0;
            for p = 1:size(background_ipm_IW_wavelet_pg, 1)
                % �����ֵ.
                if p == 1
                    bg = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end_count)) / windows_size);
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end_count);
                    continue;
                end
                bg_temp = sqrt(sum(background_ipm_IW_wavelet_pg(p, start_count:end_count)) / windows_size);
                if bg_temp > bg
                    % �滻��ֵ����.
                    best_bg = background_ipm_IW_wavelet_pg(p, start_count:end_count);
                end
                % ����ǰ����õı��ױ���.
                background_imp_IW_wavelet(j, start_count:end_count) = best_bg(1, :);
            end
        end

        start_count = start_count + windows_size;
        end_count = end_count + windows_size;
    end
end
% ==========================================================================================
% ======================== 3.�Ľ��ĵ�����С��+��˹������׿۳� ==================================
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

% �Ľ��ĵ�����С��+��˹���
background_IIW_wave_Gass = background_imp_IW_wavelet;
% ���ڸ�˹����ԸĽ��ĵ�����С�����׿۳�������н�һ���˲�.
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
% ================== �ں� ������˫����С���� �� ��������С���� ���׿۳�  =================


% =================================================================================
% ================================== ����R2 =======================================
% ������С���۳�����.
energy_windows_wave = energy - background_wavelet_adj;
% ��ǿ������С�����׿۳�.
energy_imp_iter_windows_wave = energy - background_imp_IW_wavelet;
% ��ǿ������С��+��˹������׿۳�.
energy_IIW_wave_gass = energy - background_IIW_wave_Gass;

id = zeros(1, 59);
for i = 1:59
    id(i) = str2double(strrep(filenames{i},'.txt',''));
end
% ������������ͨ��ֵ,ȡ����Ӧ�����, ����Cr��Ni��Mn��CuԪ������.
Cr_Ka = 5.41;
Ni_Ka = 7.47;
Mn_Ka = 5.895;
Cu_Ka = 8.04;

element_type = 2; % 2:Cr 3:Mn 4:Ni 5:Cu
% Cr
window_low = 149;                 %�������
window_high = 163;                %�����ҽ�
% Mn
% window_low = 165;                 %�������
% window_high = 177;                %�����ҽ�
% Cu
% window_low = 227;                 %�������
% window_high = 243;                %�����ҽ�

ag_window_low = 626;
ag_window_high = 647;
% ���ն�.
% energy_ag_expand = zeros(59, (ag_window_high - ag_window_low + 1));
% ������С��ȥ���׺���ͼ.
energy_wavelet_expand = zeros(59, (window_high - window_low + 1));
% ��ǿ������С���۳����׺���ͼ.
energy_imp_wavelet_expand = zeros(59, (window_high - window_low + 1));
% ��ǿ������С��+��˹����۳����׺���ͼ.
energy_IIW_wav_gass_expand = zeros(59, (window_high - window_low + 1));
% ԭ��ͼ.
energy_expand = zeros(59, (window_high - window_low + 1));
for i = 1:59
    for j = 1:(window_high - window_low + 1)
        % ���ն�.
        % energy_ag_expand(i,j) = energy_wavelet_gauss(i, ag_window_low+j-1);
        % ������С��ȥ���׺���ͼ.
        energy_wavelet_expand(i,j) = energy_windows_wave(i, window_low+j-1);
        % ��ǿ������С���۳����׺���ͼ.
        energy_imp_wavelet_expand(i,j) = energy_imp_iter_windows_wave(i, window_low+j-1);
        % ��ǿ������С��+��˹����۳����׺���ͼ.
        energy_IIW_wav_gass_expand(i,j) = energy_IIW_wave_gass(i, window_low+j-1);
        % ԭ��ͼ.
        energy_expand(i,j) = energy(i, window_low+j-1 );
    end
end
% ���նٷ����.
% energy_Ag_area = zeros(1,59);
% ������С��ȥ���׺���ͼ.
energy_wavelet_area = zeros(1, 59);
% ��ǿ������С���۳����׺���ͼ.
energy_imp_wave_wind_area = zeros(1, 59);
% ��ǿ������С��+��˹����۳����׺���ͼ.
energy_IIW_wav_gass_area = zeros(1, 59);
% ԭ��ͼ.
energy_area = zeros(1, 59);
for i = 1:59
    % ���ն�.
    % energy_Ag_area(i) = sum(energy_ag_expand(i,:));
    % ������С��ȥ���׺���ͼ.
    energy_wavelet_area(i) = sum(energy_wavelet_expand(i,:));
    % ��ǿ������С���۳����׺���ͼ.
    energy_imp_wave_wind_area(i) = sum(energy_imp_wavelet_expand(i,:));
    % ��ǿ������С��+��˹����۳����׺���ͼ.
    energy_IIW_wav_gass_area(i) = sum(energy_IIW_wav_gass_expand(i,:));
    % ԭ��ͼ.
    energy_area(i) = sum(energy_expand(i,:));
end

% ������С��ȥ���׿۳��������ϵ��
data = importdata('D:\Processing\Energy Analysis\Matlab\����\����.txt');
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
fprintf("������С�����׿۳��� R2=%f\n", R_2_1);

% ��ǿ������С���۳�����,�����ϵ��.
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
fprintf("��ǿ�������С�����׿۳��� R2=%f\n", R_2_2);

% ��ǿ������С��+��˹���ȥ���׿۳��������ϵ��
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
fprintf("��ǿ�������С��+��˹������׿۳��� R2=%f\n", R_2_3);


% ===============================================================================
% ============================== ���Ʊ���ͼ =======================================
figure(1);
hold on;
plot(1:1:length(energy(1,:)), energy(1, :));
plot(1:1:length(energy(1,:)), background_wavelet_adj(1,:));
plot(1:1:length(energy(1,:)), background_imp_IW_wavelet(1,:));
plot(1:1:length(energy(1,:)), background_IIW_wave_Gass(1,:));
legend("ԭ�ź�", "������С��", "��ǿ�������С��", "��ǿ�������С��+��˹���");