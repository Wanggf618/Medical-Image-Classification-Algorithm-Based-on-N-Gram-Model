clc;
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%load data %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_normal_behalf=importdata('train_normal.dat');
train_mali_behalf=importdata('train_mali.dat');
test_normal_behalf=importdata('test_normal.dat');
test_mali_behalf=importdata('test_mali.dat');
train_normal_behalf_data=train_normal_behalf.data;
train_mali_behalf_data=train_mali_behalf.data;
test_normal_behalf_data=test_normal_behalf.data;
test_mali_behalf_data=test_mali_behalf.data;
size(test_normal_behalf_data)
size(test_mali_behalf_data)
size(train_normal_behalf_data)
size(train_mali_behalf_data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%Select features by ECE%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&%
IS_0_normal=train_normal_behalf_data~=0;
num_0_normal=sum(IS_0_normal,1);
num_normal=size(train_normal_behalf_data,1);
IS_0_mali=train_mali_behalf_data~=0;
num_0_mali=sum(IS_0_mali,1);
num_mali=size(train_mali_behalf_data,1);
p_normal_t=num_0_normal./(num_0_normal+num_0_mali);
p_normal=num_normal/(num_normal+num_mali);
p_mali_t=num_0_mali./(num_0_normal+num_0_mali);
p_mali=num_normal/(num_normal+num_mali);
p_t=(num_0_normal+num_0_mali)./(num_normal+num_mali);
ECE=p_t.*(p_normal_t.*log2(max(1,p_normal_t).*p_normal.^(-1))+p_mali_t.*log2(max(1,p_mali_t).*p_mali.^(-1)));
[B,I] = sort(ECE,'descend');
num_id=ceil(length(B)*0.25)
id=I(1:num_id);
id_deal=[1:1:length(B)];
id_del=id_deal;
id_del(id)=[];
id_deal(id_del)=[];
train_normal_behalf_data=train_normal_behalf_data(:,id_deal);
train_mali_behalf_data=train_mali_behalf_data(:,id_deal);
test_normal_behalf_data=test_normal_behalf_data(:,id_deal);
test_mali_behalf_data=test_mali_behalf_data(:,id_deal);
size(test_normal_behalf_data)
size(test_mali_behalf_data)
size(train_normal_behalf_data)
size(train_mali_behalf_data)

name=string(train_normal_behalf.textdata(1,2:end));
name=name(:,id_deal)
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%libSVM%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
t=cputime
train_behalf=[train_normal_behalf_data;train_mali_behalf_data;test_normal_behalf_data;test_mali_behalf_data];
train_behalf=sparse(train_behalf);
label_normal=ones(1,size(train_normal_behalf_data,1));
label_mali=ones(1,size(train_mali_behalf_data,1))*(-1);
label_test_normal=ones(1,size(test_normal_behalf_data,1));
label_test_mali=ones(1,size(test_mali_behalf_data,1))*(-1);
label=[label_normal,label_mali,label_test_normal,label_test_mali];
label=label';
libsvmwrite('train_behalf.txt',label,train_behalf);
e=cputime-t
%%%%%%%%Enter the command in CMD%%%%%%%%%%%%%
cd D:\MatlabR2019a\toolbox\libsvm-3.25\windows
svm-scale train_behalf.txt>train_behalf_deal.txt
%Copy the file(train_behalf_deal.txt) from to F:\matlab_work\libsvm-3.23\libsvm-3.23\matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t=cputime
[lable,im]=libsvmread('train_behalf_deal.txt');
m=size(train_normal_behalf_data,1)+size(train_mali_behalf_data,1);
n=m+1;
model=svmtrain(lable(1:m),im(1:m,:),'-c 30 -g 0.01'); 
[prelabel,accuracy,decision_values]=svmpredict(lable(n:end),im(n:end,:),model);
e=cputime-t
test_num_normal=size(test_normal_behalf_data,1);
test_ID_mali=test_num_normal+1;
pre_11_y=sum(prelabel(1:test_num_normal)>0)
pre_22_y=sum(prelabel(test_ID_mali:end)<0)
name={'label','score'};
label=[label_test_normal';label_test_mali'];
data=[label,decision_values];
test_score_TF100=array2table(data,'VariableNames',name);
writetable(test_score_TF100,'RF_svm_25_1.dat','WriteVariableNames',true)
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%BP neural network%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
train_behalf=[train_normal_behalf_data;train_mali_behalf_data];
train_behalfbp=train_behalf';
label_normal=ones(1,size(train_normal_behalf_data,1));
label_mali=ones(1,size(train_mali_behalf_data,1))*(-1);
label=[label_normal,label_mali];
output=label;
test_behalf=[test_normal_behalf_data;test_mali_behalf_data];
test_behalfbp=test_behalf';
[inputn,inputps]=mapminmax(train_behalfbp);
net=newff(inputn,output,[2,4,2])

net.trainParam.epochs=100; 
net.trainParam.lr=0.001; 
net.trainParam.goal=0.00001; 
net=train(net,inputn,output);
inputn_test=mapminmax('apply',test_behalfbp,inputps);
result=sim(net,inputn_test);
pre_11_y=sum(result(1:83)>0)
pre_22_y=sum(result(84:end)<0)
name={'label','score'};
label=[label_test_normal';label_test_mali'];
data=[label,result'];
test_score_TF100=array2table(data,'VariableNames',name);
writetable(test_score_TF100,'RF_BP_25_1.dat','WriteVariableNames',true)

