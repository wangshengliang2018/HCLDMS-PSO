clear all; clc; tic;
addpath(genpath(pwd));

load Bounds;
load fbias_data;

num_particle = 40;           % number of particles 
dimension = 30;              % dimension of the variable
max_iteration = 7500;        % 
max_FES = 300000;            % num_particle * 5000Fitness Evalutions  

for i = 1:1    % problem function numbers
    fprintf('Problem =\t %d\n',i);
    func_num = i;  range = Bounds(i,:);        
    %parfor j = 1:10   % runs
    for j = 1:1   % run times
       fprintf('run =\t %d\n',j);
       [position_HCLDMS_PSO(i,j,:),value_HCLDMS_PSO(i,j),iteration_HCLDMS_PSO(i,j),max_FES_HCLDMS_PSO(i,j),Error_HCLDMS_PSO(i,j),gbestfit_HCLDMS_PSO(i,j,:)]...
                               = HCLDMS_PSO(num_particle,range,dimension,max_iteration,max_FES,func_num);                             
    end 
    if mod(i,5) == 0
       file_name = [ 'HCLDMS-PSO_',num2str(i),'_1220_30D.mat'];
       save (file_name);
    end
end    

toc;






