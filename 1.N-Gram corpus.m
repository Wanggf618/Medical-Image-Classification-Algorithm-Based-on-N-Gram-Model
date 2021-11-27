clc;
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%Malignant sample%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path='your path';
tiffile=dir([path,'*.png']);
k=length(tiffile);
mali_N=cell(1,k);
name_mali_N=cell(1,k);
for j=1:k 
	A=imread([path,tiffile(j).name]);
	mysize=size(A);
	if numel(mysize)>2
		A=rgb2gray(A);
    end
		mali_N{j}=A;
	    name_mali_N{j}=strcat(tiffile(j).name);
		name_mali_N{j}=strrep(name_mali_N{j},'.png','');
end	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%image preprocessing%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_mali_N=length(mali_N);
mali_N_deal=cell(1,num_mali_N);
for i=1:num_mali_N
    mali_N_deal{i}=mali_N{i};
	mali_N_deal{i}(:,all(mali_N_deal{i}==255,1))=[];
	mali_N_deal{i}(all(mali_N_deal{i}==255,2),:)=[];
    top=sortrows(tabulate(mali_N_deal{i}(:)),-2);
    med=min(max(top(1:11,1)),60)+5;
	mali_N_deal{i}(mali_N_deal{i}<=med)=ceil(0.75*mali_N_deal{i}(mali_N_deal{i}<=med));
    top=sortrows(tabulate(mali_N_deal{i}(:)),-2);
    med=min(max(top(1:6,1)),63);
	k=floor(16/med);
	mali_N_deal{i}(mali_N_deal{i}<=med)=ceil(k*mali_N_deal{i}(mali_N_deal{i}<=med));
	mali_N_deal{i}=wiener2(mali_N_deal{i},[2,2]);
end
%%%%%%%%%%%%%Test set20%, training set80%%%%%%%%%%%
mali_a1=[1:num_mali_N];
mali_a1=reshape(mali_a1,18,5);
test_mali_id=mali_a1(:,5)
test_mali=cell(1,18);
mali_train_N_deal=mali_N_deal;
for i=1:18
 j=test_mali_id(i); test_mali{i}=mali_train_N_deal{j};
mali_train_N_deal{j}=[];
end
name_mali_test=cell(1,18);
for i=1:18
   j=test_mali_id(i);
   name_mali_test{i}=name_mali_N{j};
end
train_mali=mali_train_N_deal;
train_mali(cellfun(@isempty,train_mali))=[];
name_mali_train=name_mali_N;
for i=1:18
   j=test_mali_id(i);
   name_mali_train{j}=[];
end
name_mali_train(cellfun(@isempty,name_mali_train))=[];
%%%%%%%%%%%%%%%%train_Malignant%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%Character matrix%%%%%%%%%%%%%%%%
mu={'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P'};
t=cputime
num_mali_train=length(train_mali); 
mali_train_letter=cell(1,num_mali_train);
str1={'A'};
str2={0.0};
mali_train_table=table;
mali_train_table.ID=str1;
mali_train_table.train=str2;
for j=1:num_mali_train
	mali_train_letter{j}=cell(size(train_mali{j}));
		for i=1:16
			A=train_mali{j}>=16*(i-1)&train_mali{j}<(16*i);
			[row,col]=find(A==1);
			mali_train_letter{j}(sub2ind(size(mali_train_letter{j}),row,col))=mu(i);
		end
    %%%%%%%%%3-Gram%%%%%%%%
	three_mali_train=im2col(mali_train_letter{j}, [1,3], 'sliding'); 	   
	three_mali_train_word=strcat(three_mali_train(1,:),three_mali_train(2,:),three_mali_train(3,:));
	three_mali_train_count=tabulate(three_mali_train_word(:));
	%%%%%%%%%2-Gram%%%%%%%%
    two_mali_train=im2col(mali_train_letter{j}, [1,2], 'sliding'); 
	two_mali_train_word=strcat(two_mali_train(1,:),two_mali_train(2,:));
	two_mali_train_count=tabulate(two_mali_train_word(:));
	%%%%%%%%%1-Gram%%%%%%%%
    one_mali_train_count=tabulate(mali_train_letter{j}(:));
	mali_train_count_part=[one_mali_train_count;two_mali_train_count;three_mali_train_count];
	third_name=strcat('frequency_',name_mali_train{j});
	three_mali_train_part=cell2table(mali_train_count_part(:,[1 2]),'VariableNames',{'ID',third_name});
	mali_train_table=outerjoin(mali_train_table,three_mali_train_part,'MergeKeys',true);
end
e=cputime-t
mali_train_table(:,'train')=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%normal sample%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path='Your path';
tiffile=dir([path,'*.*']);
tiffile(1:2)=[];
k=length(tiffile);
normal_N=cell(1,k);
name_normal_N=cell(1,k);
for j=1:k 
	A=imread([path,tiffile(j).name]);
	mysize=size(A);
	if numel(mysize)>2
		A=rgb2gray(A);
    end
		normal_N{j}=A;
	    name_normal_N{j}=strcat(tiffile(j).name);
		name_normal_N{j}=strrep(name_normal_N{j},'.png','');
		name_normal_N{j}=strrep(name_normal_N{j},'.jpg','');
end	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%image preprocessing%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_normal_N=length(normal_N);
normal_N_deal=cell(1,num_normal_N);
for i=1:num_normal_N
    normal_N_deal{i}=normal_N{i};
	normal_N_deal{i}(:,all(normal_N_deal{i}==255,1))=[];
	normal_N_deal{i}(all(normal_N_deal{i}==255,2),:)=[];
    top=sortrows(tabulate(normal_N_deal{i}(:)),-2);
    med=min(max(top(1:11,1)),60)+5;
	normal_N_deal{i}(normal_N_deal{i}<=med)=ceil(0.75*normal_N_deal{i}(normal_N_deal{i}<=med));
    top=sortrows(tabulate(normal_N_deal{i}(:)),-2);
    med=min(max(top(1:6,1)),63);
	k=floor(16/med);
	normal_N_deal{i}(normal_N_deal{i}<=med)=ceil(k*normal_N_deal{i}(normal_N_deal{i}<=med));
	normal_N_deal{i}=wiener2(normal_N_deal{i},[2,2]);
end
%%%%%%%%%%%%%Test set20%, training set80%%%%%%%%%%%
normal_a1=[1:num_normal_N];
normal_a1=reshape(normal_a1,83,5);
test_normal_id=normal_a1(:,5)
test_normal=cell(1,83);
test_N_normal=normal_N_deal;
for i=1:83
 j=test_normal_id(i);
 test_normal{i}=test_N_normal{j};
 test_N_normal{j}=[];
end

name_normal_test=cell(1,83);
for i=1:83
 j=test_normal_id(i);
 name_normal_test{i}=name_normal_N{j};
end
train_normal=test_N_normal;
train_normal(cellfun(@isempty,train_normal))=[];
name_normal_train=name_normal_N;
for i=1:83
   j=test_normal_id(i);
   name_normal_train{j}=[];
end
name_normal_train(cellfun(@isempty,name_normal_train))=[];
%%%%%%%%%%%%%%%%train_normal%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%Character matrix%%%%%%%%%%%%%%%%
t=cputime
num_normal_train=length(train_normal); 
normal_train_letter=cell(1,num_normal_train);
str1={'A'};
str2={0.0};
normal_train_table=table;
normal_train_table.ID=str1;
normal_train_table.train=str2;
for j=1:num_normal_train
	normal_train_letter{j}=cell(size(train_normal{j}));
		for i=1:16
			A=train_normal{j}>=16*(i-1)&train_normal{j}<(16*i);
			[row,col]=find(A==1);
			normal_train_letter{j}(sub2ind(size(normal_train_letter{j}),row,col))=mu(i);
		end
	three_normal_train=im2col(normal_train_letter{j}, [1,3], 'sliding'); 
	three_normal_train_word=strcat(three_normal_train(1,:),three_normal_train(2,:),three_normal_train(3,:));
	three_normal_train_count=tabulate(three_normal_train_word(:));
	two_normal_train=im2col(normal_train_letter{j}, [1,2], 'sliding'); 
	two_normal_train_word=strcat(two_normal_train(1,:),two_normal_train(2,:));
	two_normal_train_count=tabulate(two_normal_train_word(:));
	one_normal_train_count=tabulate(normal_train_letter{j}(:));
	normal_train_count_part=[one_normal_train_count;two_normal_train_count;three_normal_train_count];
	third_name=strcat('frequency_',name_normal_train{j});
	three_normal_train_part=cell2table(normal_train_count_part(:,[1 2]),'VariableNames',{'ID',third_name});
	normal_train_table=outerjoin(normal_train_table,three_normal_train_part,'MergeKeys',true);
end
normal_train_table(:,'train')=[];
e=cputime-t
%%Extract the union of all the training set corpus%%
normal_mali_table=outerjoin(normal_train_table,mali_train_table,'MergeKeys',true);
name=table2array(normal_mali_table(:,1));
sample=normal_mali_table.Properties.VariableNames(2:end);
normal_mali_table_tem=table2array(normal_mali_table(:,2:end));
[x,y]=find(isnan(normal_mali_table_tem)==1);
normal_mali_table_tem(sub2ind(size(normal_mali_table_tem),x,y))=0;
sum_sample=sum(normal_mali_table_tem,1);
normal_mali_table_new=normal_mali_table_tem./sum_sample;
normal_mali_table_array=normal_mali_table_new';
normal_mali_table_t=array2table(normal_mali_table_array,'VariableNames',name,'RowNames',sample);
writetable(normal_mali_table_t,'train_normal_mali_table5.dat','WriteRowNames',true)
%%%%%%%%%%%%%%%%test_mali%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%Character matrix%%%%%%%%%%%%%%
num_mali_test=length(test_mali); 
mali_test_letter=cell(1,num_mali_test);
str1={'A'};
str2={0.0};
mali_test_table=table;
mali_test_table.ID=str1;
mali_test_table.test=str2;
for j=1:num_mali_test
 %j=1
	mali_test_letter{j}=cell(size(test_mali{j}));
		for i=1:16
			A=test_mali{j}>=16*(i-1)&test_mali{j}<(16*i);
			[row,col]=find(A==1);
			mali_test_letter{j}(sub2ind(size(mali_test_letter{j}),row,col))=mu(i);
		end
	%%%%%%%%%3-Gram%%%%%%%%
	three_mali_test=im2col(mali_test_letter{j}, [1,3], 'sliding');
	three_mali_test_word=strcat(three_mali_test(1,:),three_mali_test(2,:),three_mali_test(3,:));
	three_mali_test_count=tabulate(three_mali_test_word(:));
	%%%%%%%%%2-Gram%%%%%%%%
	two_mali_test=im2col(mali_test_letter{j}, [1,2], 'sliding');
	two_mali_test_word=strcat(two_mali_test(1,:),two_mali_test(2,:));
	two_mali_test_count=tabulate(two_mali_test_word(:));
	%%%%%%%%%1-Gram%%%%%%%%
	one_mali_test_count=tabulate(mali_test_letter{j}(:));
	mali_test_count_part=[one_mali_test_count;two_mali_test_count;three_mali_test_count];
	third_name=strcat('frequency_',name_mali_test{j});
	three_mali_test_part=cell2table(mali_test_count_part(:,[1 2]),'VariableNames',{'ID',third_name});
	mali_test_table=outerjoin(mali_test_table,three_mali_test_part,'MergeKeys',true);
end
mali_test_table(:,'test')=[];
writetable(mali_test_table,'test_mali_table.dat','WriteRowNames',true)
%%%%%%%%%%%%%%%%test_normal%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%Character matrix%%%%%%%%%%%%%%
num_normal_test=length(test_normal); 
normal_test_letter=cell(1,num_normal_test);
str1={'A'};
str2={0.0};
normal_test_table=table;
normal_test_table.ID=str1;
normal_test_table.test=str2;
for j=1:num_normal_test
	normal_test_letter{j}=cell(size(test_normal{j}));
		for i=1:16
			A=test_normal{j}>=16*(i-1)&test_normal{j}<(16*i);
			[row,col]=find(A==1);
			normal_test_letter{j}(sub2ind(size(normal_test_letter{j}),row,col))=mu(i);
		end
	%%%%%%%%%3-Gram%%%%%%%%
	three_normal_test=im2col(normal_test_letter{j}, [1,3], 'sliding');
	three_normal_test_word=strcat(three_normal_test(1,:),three_normal_test(2,:),three_normal_test(3,:));
	three_normal_test_count=tabulate(three_normal_test_word(:));
	%%%%%%%%%2-Gram%%%%%%%%
	two_normal_test=im2col(normal_test_letter{j}, [1,2], 'sliding');
	two_normal_test_word=strcat(two_normal_test(1,:),two_normal_test(2,:));
	two_normal_test_count=tabulate(two_normal_test_word(:));
	%%%%%%%%%1-Gram%%%%%%%%
	one_normal_test_count=tabulate(normal_test_letter{j}(:));
	normal_test_count_part=[one_normal_test_count;two_normal_test_count;three_normal_test_count];
	third_name=strcat('frequency_',name_normal_test{j});
	three_normal_test_part=cell2table(normal_test_count_part(:,[1 2]),'VariableNames',{'ID',third_name});
	normal_test_table=outerjoin(normal_test_table,three_normal_test_part,'MergeKeys',true);
end

normal_test_table(:,'test')=[];
writetable(normal_test_table,'test_normal_table.dat','WriteRowNames',true)









