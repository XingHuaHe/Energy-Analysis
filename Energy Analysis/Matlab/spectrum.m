% Description:
%       ������Ҫ�������������ݽ���Ԥ����
%       ��1����ȡ�����ļ�
%       ��2����ȡexcel�ļ�-��Ʒ���ƶ�Ӧ��
% Autor:
%   XingHua.He
% History:
%   2020.12.27

clc; clear; close all;
% =================================================================================================
% ��ȡ��׼������.
[stander_contents, stander_tfiles, ~] = xlsread('D:\Processing\Energy Analysis\Matlab\����ʵ������\stander contents.xls', 1, 'B2:CX60');
% ��ȡ�ļ����ơ�����Ʒ���ӳ�� Excel ���.
[~, samples, ~] = xlsread('D:\Processing\Energy Analysis\Matlab\����ʵ������\spectrum datas\202009221558543521ls.xlsx', 'B2:B60');
[~, tfiles, ~] = xlsread('D:\Processing\Energy Analysis\Matlab\����ʵ������\spectrum datas\202009221558543521ls.xlsx', 'D2:D60');

% =================================================================================================
% ��ȡ�����ļ�����.
fileFolder = fullfile('D:\Processing\Energy Analysis\Matlab\����ʵ������\spectrum datas');
dirOutput = dir(fullfile(fileFolder,'*.txt'));
filenames = {dirOutput.name};

spectrums = zeros(59, 2148);
% Load all .txt files.
for i = 1:59
    filename = filenames{i}(1:end-4);
    energy_buff = importdata(strcat('D:\Processing\Energy Analysis\Matlab\����ʵ������\spectrum datas\' , filenames{i}));
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