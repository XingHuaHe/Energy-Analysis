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

% 读取路径下txt文件名
fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\土壤\土壤谱图\新建文件夹');
dirOutput = dir(fullfile(fileFolder,'*.txt'));
filenames = {dirOutput.name};
energy = zeros(59,2048);
energy_x = 1:2048;
%读取每一个txt文件
for i = 1:59
    energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\土壤\土壤谱图\新建文件夹\' , filenames{i}));
    if i == 38
        energy(i,:) = energy_buff(:,1)';
    else
        energy(i,:) = energy_buff(:,2)';
    end
end

%生成高斯卷积核
dimension = 5;    %卷积核大小
sigma = 1;        %卷积核参数
kernel = zeros(1, dimension);
for loop = 1:dimension
    x_axis = -floor(dimension/2) + loop - 1;
    kernel(loop) = 1 / sigma * exp(-x_axis^2 / (2*sigma^2));
end
kernel = kernel ./ sum(kernel);

%定义迭代次数
iteration = 500;

%迭代高斯卷积滤波去本底
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

%定义Cr，Ni，Mn，Cu元素能量
Cr_Ka = 5.41;
Ni_Ka = 7.47;
Mn_Ka = 5.895;
Cu_Ka = 8.04;

%根据能量反算通道值,取出对应峰面积
%a=0.0337289156626506;
%b=-0.0756084337349394;
window_low = 149;                 %卡窗左界
window_high = 163;                %卡窗右界
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

%根据本底扣除后计数峰面积和实际样品含量拟合直线，算相关系数
data = importdata('D:\Processing\Energy Analysis\Matlab\土壤\数据.txt');
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
