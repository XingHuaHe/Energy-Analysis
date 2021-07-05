% Description:
%       本实验采用人工蜂群优化算法优化高斯卷积.
%   
% Autor:
%   XingHua.He
% History:
%   2020.11.18

clc; clear; close all;
addpath('./Artificial Bee Colony');
% =================================================================================================
% ========================================== 读取数据 =======================================
fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\土壤\土壤谱图\新建文件夹');
dirOutput = dir(fullfile(fileFolder,'*.txt'));
filenames = {dirOutput.name};
energy = zeros(1,2048);
energy_x = 1:2048;
% Load all .txt files.
for i = 1:59
    energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\土壤\土壤谱图\新建文件夹\' , filenames{i}));
    if i == 38
        energy(i,:) = energy_buff(:,1)';
    else
        energy(i,:) = energy_buff(:,2)';
    end
    break;
end
% =================================================================================================
% ================================== ABC优化 =======================================
for i = 1:3
sigma = abc_fun();
% sigma
end

% =================================================================================================
% ========================================== 生成高斯卷积核 =======================================
% Conv-kernel size.
dimension = 5;
% Conv-kernel parameter.
% sigma = 1;
kernel = zeros(1, dimension);
for loop = 1:dimension
    x_axis = -floor(dimension/2) + loop - 1;
    kernel(loop) = 1 / sigma * exp(-x_axis^2 / (2*sigma^2));
end
kernel = kernel ./ sum(kernel);
% =================================================================================================
% ================================== 利用高斯卷积本底扣除 =======================================
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
% ================================== 绘图 =======================================
figure(1);
hold on;
plot(1:1:2048, energy);
plot(1:1:2048, background_guass);
