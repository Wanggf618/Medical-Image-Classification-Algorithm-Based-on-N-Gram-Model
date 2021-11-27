clc;
clear;
label_matrix=[];
scores=[];
num_time=cell(1,4)
time_file=[25 50 75 100];
tf_rd={'TF\SVM','TF\BP','TF.RF\SVM','TF.RF\BP'};
tit={'TF-SVM','TF-BP','TF.RF-SVM','TF.RF-BP'};

for k=1:4
subplot(2,2,k)
for j=1:4
path=strcat('Your path',char(tf_rd(k)),'\',num2str(time_file(j)),'\');
tiffile=dir([path,'*.dat']);
for i=1:5
datfile=importdata([path,tiffile(i).name])
label_matrix=[label_matrix;datfile.data(:,1)];
scores=[scores;datfile.data(:,2)];
end
[Xroc_sum,Yroc_sum,~,AUC_sum]=perfcurve(label_matrix, scores,1);
plot(Xroc_sum, Yroc_sum, 'LineWidth', 1)
axis equal
hold on
num_time{j}=sprintf('%s%%(AUC=%.3f)',num2str(time_file(j)),AUC_sum)
end
titK=sprintf('ROC for Classification by %s',char(tit(k)));
legend(num_time)
xlabel('False positive rate') 
ylabel('True positive rate')
title(titK)
end



