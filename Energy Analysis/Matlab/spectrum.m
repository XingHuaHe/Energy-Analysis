% Description:
%       程序主要操作——对数据进行预处理：
%       （1）读取光谱文件
%       （2）读取excel文件-样品名称对应表
% Autor:
%   XingHua.He
% History:
%   2020.12.27

clc; clear; close all;
% =================================================================================================
% 读取标准含量表.
[stander_contents, stander_tfiles, ~] = xlsread('D:\Processing\Energy Analysis\Matlab\土壤实验数据\stander contents.xls', 1, 'B2:CX60');
% 读取文件名称——样品编号映射 Excel 表格.
[~, samples, ~] = xlsread('D:\Processing\Energy Analysis\Matlab\土壤实验数据\spectrum datas\202009221558543521ls.xlsx', 'B2:B60');
[~, tfiles, ~] = xlsread('D:\Processing\Energy Analysis\Matlab\土壤实验数据\spectrum datas\202009221558543521ls.xlsx', 'D2:D60');

% =================================================================================================
% 获取测试文件名称.
fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\土壤实验数据\spectrum datas');
dirOutput = dir(fullfile(fileFolder,'*.txt'));
filenames = {dirOutput.name};

spectrums = zeros(59, 2148);
% Load all .txt files.
for i = 1:59
    filename = filenames{i}(1:end-4);
    energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\土壤实验数据\spectrum datas\' , filenames{i}));
    % save content.
    for j = 1:59
        if strcmp(tfiles{j}, filename)
            % tfiles{j} == filename
            for k = 1:59
               if strcmp(stander_tfiles{k}, samples{j}(1:end-2)) || strcmp(stander_tfiles{k}, samples{j}(1:end-3))
                   spectrums(i, 2049:end) = stander_contents(k, :);
               end
            end
            
        end
    end
    % save spectrum datas.
    spectrums(i,1:2048) = energy_buff(:,1)';
end
save('soil.mat', 'spectrums');