% Description:
%       ��ʵ������
%       ��1������С�����׿۳�
%   	��2��������С�����׿۳�
%       ��3��������С��+������˹������׿۳�
%       ��4��������˹���
%       ��5���Ľ�������С�����׿۳������붯����
%       ��6���Ľ�������С��+��˹������׿۳������붯����
% Autor:
%   XingHua.He
% History:
%   2020.12.27

clc; clear; close all;
% =================================================================================================
% ========================================== ��ȡ���� =======================================
% fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���');
% dirOutput = dir(fullfile(fileFolder,'*.txt'));
% filenames = {dirOutput.name};
% energy = zeros(59,2048);
% energy_x = 1:2048;
% % Load all .txt files.
% for i = 1:59
%     energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���\' , filenames{i}));
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
% ======================================== 1.����С�����׿۳� =====================================
% energy = energy(:,1:1300);
background_wavelet = energy;

for j = 1:size(background_wavelet, 1)
    [c,l] = wavedec(background_wavelet(j, :) ,6,'sym4');%��߶�һά��ɢС�����ع���1-7��ƽ�ϵ�� db5
    a7 = wrcoef('a',c,l,'sym4',6);

    for iloop = 1:20
        for i = 1:length(background_wavelet(j, :))
            if(background_wavelet(j, i) > a7(i) && a7(i) > 0)
                background_wavelet(j, i)=a7(i);
            end
        end
        [c,l] = wavedec(background_wavelet(j, :), 6, 'sym4');%�ع���1-7��ƽ�ϵ��
        a7 = wrcoef('a', c, l, 'sym4', 6);
    end
end
% =================================================================================================
% ============================== 2.������С�����׿۳� ============================================
background_wavelet_adj = energy;
% ��ѡС����.
wavelet_package = {'sym4', 'bior6.8'};%{'db5', 'db8', 'db30', 'sym4', 'bior6.8'};
% ����С.
windows_size = 25;

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

% =================================================================================================
% ====================== 3.������С��+������˹������׿۳� ============================================
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

background_wavelet_gauss = background_wavelet_adj;
for z = 1:size(background_wavelet_gauss, 1)
    guass = background_wavelet_gauss(z, :);
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
    background_wavelet_gauss(z, :) = guass(1, :);
end
% =================================================================================================
% ============================== 4.������˹��� =====================================
background_gauss = energy;
for z = 1:size(background_gauss, 1)
    guass = background_gauss(z, :);
    for i = 1:500
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
    background_gauss(z, :) = guass(1, :);
end

% =================================================================================================
% ===================== 5.�Ľ�������С�����׿۳������붯���� ================================
background_imp_IW_wavelet = energy;
% ��ѡС����.
wavelet_package = {'sym4', 'bior6.8'}; % {'db5', 'db8', 'db30', 'sym4', 'bior6.8'};
% ����С.
windows_size = 50;
% �ֲ��仯������.
windows_consider = 10;
% ����.
momentum = 1;

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
            % energy_local = zeros(1, windows_size*(windows_consider+w));
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
                            background_wavelet_temp2(i) = momentum*a7(i) + (1-momentum)*background_wavelet_temp2(i);
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
            % energy_local = zeros(1, windows_size*(2*windows_consider+1));
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
                            background_wavelet_temp2(i) =momentum*a7(i) + (1-momentum)*background_wavelet_temp2(i);
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
            % energy_local = zeros(1, windows_size*(windows_consider)+length(background_wavelet_temp(1, start_count:end)));
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
                            background_wavelet_temp2(i) = momentum*a7(i) + (1-momentum)*background_wavelet_temp2(i);
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
% =================================================================================================
% ===================== 6.�Ľ�������С��+��˹������׿۳������붯���� ================================
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

% Iter number.
iteration = 100;
% ����.
momentum = 1;

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
                guass(1,j) = momentum*background_buff(j) + (1-momentum)*guass(1,j);
            else
                guass(1,j) = guass(1,j);
            end
        end
    end
    background_IIW_wave_Gass(z, :) = guass(1, :);
end

% ===========================================================================
% ======================= ����ͼ���� =====================================
% save('./Result save/energy.mat', 'energy');
% save('./Result save/background_wavelet.mat', 'background_wavelet');
% save('./Result save/background_IIW_wave_Gass.mat', 'background_IIW_wave_Gass');

% =================================================================================================
% ================================== ����R2 =======================================
% С���۳�����.
energy_wavelet = energy - background_wavelet;
% ��С���۳�����.
energy_wavelet_windows = energy - background_wavelet_adj;
% ��С�� + ��˹���.
energy_wavelet_gauss = energy - background_wavelet_gauss;
% ��˹���.
energy_gauss = energy - background_gauss;
% �Ľ�������С�����׿۳������붯����
energy_IIW_wave = energy - background_imp_IW_wavelet;
% �Ľ�������С��+��˹������׿۳������붯����
energy_IIW_wave_gass = energy - background_IIW_wave_Gass;

% id = zeros(1, 59);
% for i = 1:59
%     id(i) = str2double(strrep(filenames{i},'.txt',''));
% end
% ������������ͨ��ֵ,ȡ����Ӧ�����, ����Cr��Ni��Mn��CuԪ������.
Cr_Ka = 5.41;
Ni_Ka = 7.47;
Mn_Ka = 5.895;
Cu_Ka = 8.04;
%a=0.0337289156626506;
%b=-0.0756084337349394;
%axis = floor((Cr_Ka - b)/a);

element_type = 4; % 1:Cr 2:Mn 3:Ni 4:Cu
% data = importdata('D:\Processing\Energy Analysis\Matlab\����\����.txt');
data = zeros(59, 4);
data(:, 1:2) = contents(:, 24:25);
data(:, 3:4) = contents(:, 28:29);

% Cr
% window_low = 149;                 %�������
% window_high = 163;                %�����ҽ�
% Mn
% window_low = 169;                 %�������
% window_high = 173;                %�����ҽ�
% window_low = 157;                 %�������
% window_high = 170;                %�����ҽ�
% window_low = 160;                 %�������
% window_high = 173;                %�����ҽ�
% Cu
% window_low = 230;                 %�������
% window_high = 247;                %�����ҽ�
window_low = 230;                 %�������
window_high = 240;                %�����ҽ�

ag_window_low = 572;
ag_window_high = 624;
% ���ն�.
energy_ag_expand = zeros(59, (ag_window_high - ag_window_low + 1));
% ����С��ȥ���׺���ͼ.
energy_wavelet_expand = zeros(59, (window_high - window_low + 1));
% ��С���۳����׺���ͼ.
energy_wavelet_wind_expand = zeros(59, (window_high - window_low + 1));
% ��С�� + ��˹���.
energy_wavelet_gauss_expand = zeros(59, (window_high - window_low + 1));
% ��˹���.
energy_gauss_expand = zeros(59, (window_high - window_low + 1));
% �Ľ�������С�����׿۳������붯����
energy_IIW_wave_expand = zeros(59, (window_high - window_low + 1));
% �Ľ�������С��+��˹������׿۳������붯����
energy_IIW_wave_gass_expand = zeros(59, (window_high - window_low + 1));
% ԭ��ͼ.
energy_expand = zeros(59, (window_high - window_low + 1));
for i = 1:59
    for j = 1:(window_high - window_low + 1)
        % ���ն�.
        energy_ag_expand(i,j) = energy_IIW_wave_gass(i, ag_window_low+j-1);
        % ����С��ȥ���׺���ͼ.
        energy_wavelet_expand(i,j) = energy_wavelet(i,window_low+j-1);
        % ��С���۳����׺���ͼ.
        energy_wavelet_wind_expand(i,j) = energy_wavelet_windows(i, window_low+j-1);
        % ��С�� + ��˹���.
        energy_wavelet_gauss_expand(i,j) = energy_wavelet_gauss(i, window_low+j-1);
        % ��˹���.
        energy_gauss_expand(i,j) = energy_gauss(i, window_low+j-1);
        % �Ľ�������С�����׿۳������붯����
        energy_IIW_wave_expand(i,j) = energy_IIW_wave(i, window_low+j-1);
        % �Ľ�������С��+��˹������׿۳������붯����
        energy_IIW_wave_gass_expand(i,j) = energy_IIW_wave_gass(i, window_low+j-1);
        % ԭ��ͼ.
        energy_expand(i,j) = energy(i, window_low+j-1 );
    end
end
% ���նٷ����.
energy_Ag_area = zeros(1,59);
% ����С��ȥ���׺���ͼ.
energy_wavelet_area = zeros(1,59);
% ��С���۳����׺���ͼ.
energy_wavelet_wind_area = zeros(1,59);
% ��С�� + ��˹���.
energy_wavelet_gauss_area = zeros(1,59);
% ��˹���.
energy_gauss_area = zeros(1,59);
% �Ľ�������С�����׿۳������붯����
energy_IIW_wave_area = zeros(1,59);
% �Ľ�������С��+��˹������׿۳������붯����
energy_IIW_wave_gass_area = zeros(1,59);
% ԭ��ͼ.
energy_area = zeros(1,59);
for i = 1:59
    % ���ն�.
    energy_Ag_area(i) = sum(energy_ag_expand(i,:));
    % ����С��ȥ���׺���ͼ.
    energy_wavelet_area(i) = sum(energy_wavelet_expand(i,:));
    % ��С���۳����׺���ͼ.
    energy_wavelet_wind_area(i) = sum(energy_wavelet_wind_expand(i,:));
    % ��С�� + ��˹���.
    energy_wavelet_gauss_area(i) = sum(energy_wavelet_gauss_expand(i,:));
    % ��˹���.
    energy_gauss_area(i) = sum(energy_gauss_expand(i,:));
    % �Ľ�������С�����׿۳������붯����
    energy_IIW_wave_area(i) = sum(energy_IIW_wave_expand(i,:));
    % �Ľ�������С��+��˹������׿۳������붯����
    energy_IIW_wave_gass_area(i) = sum(energy_IIW_wave_gass_expand(i,:));
    % energy_IIW_wave_gass_area(i) = sum(energy_IIW_wave_gass_expand(i,:)) / energy_Ag_area(i);
    % ԭ��ͼ.
    energy_area(i) = sum(energy_expand(i,:));
end

% С�����׿۳��������ϵ��
contents_wavelet = zeros(59,2);
for i = 1:59
    contents_wavelet(i,1) = energy_wavelet_area(i);
    contents_wavelet(i,2) = data(i, element_type);
%     for j = 1:59
%         if data(j,1) == id(i)
%             contents_wavelet(i,2) = data(j, element_type);
%             break
%         end
%     end
end
R1 = corrcoef(contents_wavelet);
R_2_1 = (R1(1,2)^2);
fprintf("С�����׿۳��� R2=%f\n", R_2_1);
save('./Result save/wavelet.mat', 'contents_wavelet');

% ��С���۳�����,�����ϵ��.
contents_wavelet_windows = zeros(59,2);
for i = 1:59
    contents_wavelet_windows(i,1) = energy_wavelet_wind_area(i);
    contents_wavelet_windows(i,2) = data(i, element_type);
%     for j = 1:59
%         if data(j,1) == id(i)
%             contents_wavelet_windows(i,2) = data(j, element_type);
%             break
%         end
%     end
end
R2 = corrcoef(contents_wavelet_windows);
R_2_2 = (R2(1,2)^2);
fprintf("��С���� R2=%f\n", R_2_2);
save('./Result save/windows_wavelet.mat', 'contents_wavelet_windows');

% ��С�� + ��˹����������ϵ��
contents_wavelet_gauss = zeros(59,2);
for i = 1:59
    contents_wavelet_gauss(i,1) = energy_wavelet_gauss_area(i);
    contents_wavelet_gauss(i,2) = data(i, element_type);
%     for j = 1:59
%         if data(j,1) == id(i)
%             contents_wavelet_gauss(i,2) = data(j, element_type);
%             break
%         end
%     end
end
R3 = corrcoef(contents_wavelet_gauss);
R_2_3 = (R3(1,2)^2);
fprintf("С��+��˹�۳��� R2=%f\n", R_2_3);
save('./Result save/wavelet_gass.mat', 'contents_wavelet_gauss');

% ��˹����������ϵ��.
contents_gauss = zeros(59,2);
for i = 1:59
    contents_gauss(i,1) = energy_gauss_area(i);
    contents_gauss(i,2) = data(i, element_type);
%     for j = 1:59
%         if data(j,1) == id(i)
%             contents_gauss(i,2) = data(j, element_type);
%             break
%         end
%     end
end
R4 = corrcoef(contents_gauss);
R_2_4 = (R4(1,2)^2);
fprintf("��˹�۳��� R2=%f\n", R_2_4);
save('./Result save/gass.mat', 'contents_gauss');

% �Ľ�������С�����׿۳������붯����,�����ϵ��.
contents_IIW_wave = zeros(59,2);
for i = 1:59
    contents_IIW_wave(i,1) = energy_IIW_wave_area(i);
    contents_IIW_wave(i,2) = data(i, element_type);
%     for j = 1:59
%         if data(j,1) == id(i)
%             contents_IIW_wave(i,2) = data(j, element_type);
%             break
%         end
%     end
end
R5 = corrcoef(contents_IIW_wave);
R_2_5 = (R5(1,2)^2);
fprintf("�Ľ�������С���۳������붯������ R2=%f\n", R_2_5);
save('./Result save/IIW_wave.mat', 'contents_IIW_wave');

% �Ľ�������С��+��˹������׿۳������붯����,�����ϵ��
contents_IIW_wave_gass = zeros(59,2);
for i = 1:59
    contents_IIW_wave_gass(i,1) = energy_IIW_wave_gass_area(i);
    contents_IIW_wave_gass(i,2) = data(i, element_type);
%     for j = 1:59
%         if data(j,1) == id(i)
%             contents_IIW_wave_gass(i,2) = data(j, element_type);
%             break
%         end
%     end
end
R6 = corrcoef(contents_IIW_wave_gass);
R_2_6 = (R6(1,2)^2);
fprintf("�Ľ�������С��+��˹����۳������붯������ R2=%f\n", R_2_6);
% save('./Result save/IIW_wave_gass.mat', 'contents_IIW_wave_gass');
save('./Result save/IIW_wave_gass_not_mom.mat', 'contents_IIW_wave_gass');

% ԭ�źţ������ϵ��
contents = zeros(59,2);
for i = 1:59
    contents(i,1) = energy_area(i);
    contents(i,2) = data(i, element_type);
%     for j = 1:59
%         if data(j,1) == id(i)
%             contents(i,2) = data(j, element_type);
%             break
%         end
%     end
end
R = corrcoef(contents);
R_2 = (R(1,2)^2);
fprintf("δ���б��׿۳��� R2=%f\n", R_2);
save('./Result save/orgin.mat', 'contents');

% =================================================================================================
% ================================== ��ͼ =======================================
figure(1);
hold on;
plot(1:1:length(energy(1,:)), energy(2,:));
plot(1:1:length(energy(1,:)), background_wavelet_gauss(2,:));
plot(1:1:length(energy(1,:)), background_IIW_wave_Gass(2,:));
% plot(1:1:2048, background_wavelet_adj(1,:));
% plot(1:1:length(energy(1,:)), energy_wavelet_gauss(2,:));
legend('orgin', 'wave and gass', 'improve wave and gass');

% =================================================================================================
% ============================ �������׿۳�����ͼ�������������� ====================================
% fprintf("����-�Ľ�������С���۳������붯����-��ͼ\n");
% spectrums = zeros(59,2049);
% spectrums(:, 1:2048) = energy_IIW_wave_gass(:,1:end);
% spectrums(:, 2049) = data(:, element_type);
% save('soil_bg.mat', 'spectrums');

features = zeros(59, 13);
% Cucontent = zeros(59, 1);
for i = 1:size(features, 1)
    % V 4.51
    features(i, 1) = sum(energy_IIW_wave_gass(i, 139:148));
    % Cr 5.41
    features(i, 2) = sum(energy_IIW_wave_gass(i, 149:163));
    % Mn 5.895
    features(i, 3) = sum(energy_IIW_wave_gass(i, 168:175));
    % Ni 7.47
    features(i, 4) = sum(energy_IIW_wave_gass(i, 211:220));
    % Cu 8.04
    features(i, 5) = sum(energy_IIW_wave_gass(i, 227:245));
    % Zn 8.63
    features(i, 6) = sum(energy_IIW_wave_gass(i, 245:255));
    % Pb 10.549
    features(i, 7) = sum(energy_IIW_wave_gass(i, 297:311)); 
    % As 11.729
    features(i, 8) = sum(energy_IIW_wave_gass(i, 332:345)); 
    % Bi 10.84
    features(i, 9) = sum(energy_IIW_wave_gass(i, 311:318)); 
    % Tl 10.266
    features(i, 10) = sum(energy_IIW_wave_gass(i, 290:300)); 
end
% 1:Cr 2:Mn 3:Ni 4:Cu
features(:, 11) = data(:, 1);
features(:, 12) = data(:, 2);
features(:, 13) = data(:, 4);
% save('features.mat', 'features');
% save('Cucontent.mat', 'Cucontent');