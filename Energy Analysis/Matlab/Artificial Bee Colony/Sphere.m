%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA114
% Project Title: Implementation of Artificial Bee Colony in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function z=Sphere(x)

% ��ȡ·����txt�ļ���
fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���');
dirOutput = dir(fullfile(fileFolder,'*.txt'));
filenames = {dirOutput.name};
energy = zeros(59,2048);
energy_x = 1:2048;
%��ȡÿһ��txt�ļ�
for i = 1:59
    energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\����\������ͼ\�½��ļ���\' , filenames{i}));
    if i == 38
        energy(i,:) = energy_buff(:,1)';
    else
        energy(i,:) = energy_buff(:,2)';
    end
end

%���ɸ�˹�����
dimension = 5;    %����˴�С
sigma = 1;        %����˲���
kernel = zeros(1, dimension);
for loop = 1:dimension
    x_axis = -floor(dimension/2) + loop - 1;
    kernel(loop) = 1 / sigma * exp(-x_axis^2 / (2*sigma^2));
end
kernel = kernel ./ sum(kernel);

%�����������
iteration = 500;

%������˹����˲�ȥ����
background = energy;
%background = zeros(59, 2050);
for k = 1:59
    for i = 1:iteration
        background_buff = conv(kernel, background(k, :));
        background_buff = background_buff((floor(dimension/2)+1):(floor(dimension/2)+1+2047));
        for j = 1:2048
            if background_buff(j) < background(k,j)
                background(k,j) = background_buff(j);
            else
                background(k,j) = background(k,j);
            end
        end
    end
end

energy_new = energy - background;

id = zeros(1, 59);
for i = 1:59
    id(i) = str2double(strrep(filenames{i},'.txt',''));
end

%����Cr��Ni��Mn��CuԪ������
Cr_Ka = 5.41;
Ni_Ka = 7.47;
Mn_Ka = 5.895;
Cu_Ka = 8.04;

%������������ͨ��ֵ,ȡ����Ӧ�����
%a=0.0337289156626506;
%b=-0.0756084337349394;
window_low = 149;                 %�������
window_high = 163;                %�����ҽ�
%axis = floor((Cr_Ka - b)/a);
energy_expand_new = zeros(59, (window_high - window_low + 1));
energy_expand = zeros(59, (window_high - window_low + 1));
for i = 1:59
    for j = 1:(window_high - window_low+1)
        energy_expand_new(i,j) = energy_new(i,window_low+j-1 );
        energy_expand(i,j) = energy(i,window_low+j-1 );
    end
end

energy_area_new = zeros(1,59);
for i = 1:59
    energy_area_new(i) = sum(energy_expand_new(i,:));
end

%���ݱ��׿۳�������������ʵ����Ʒ�������ֱ�ߣ������ϵ��
data = importdata('D:\Processing\Energy Analysis\Matlab\����\����.txt');
contents = zeros(59,2);
for i = 1:59
    contents(i,1) = energy_area_new(i);
    for j = 1:59
        if data(j,1) == id(i)
            contents(i,2) = data(j,2);
            break
        end
    end
end

R = corrcoef(contents);
R_2 = (R(1,2)^2);

z = 1 - R_2;
end
