function [position,value,iteration,max_FES,Error,gbestfit]= HCLDMS_PSO(num_particle,range,dimension,max_iteration,max_FES,func_num) 
% Input：num_particle is the number of the population; 
%        range is the boundary of variable; dimension is the problem dimension;
%        max_iteration is the maximal run times; 
%        max_FES is the maximal fitness evalution times; 
%        func_num is the optimization problem function number;
% Output：position is the best position of the solution;
%         value is the optimum fitness value;
%         iteration is the iteration run times; 
%         max_FES is the fitness evalution times; 
%         Error is the deviation between the optimization result and the truth value
%        gbestfit is the difference between the population optimal fitness value and the truth value in the evolutionary process
% by wangshengliang,IGG,CAS, 2019/12/22 First revision； 
% Paper: Heterogeneous comprehensive learning and dynamic multi-swarm particle swarm optimizer with two mutation operators

rand('state',sum(100*clock));
load fbias_data;

check_vel = [];
check_vel1 = [];
check_vel2 = [];
% *********************** Parameter set ***********************************
num_g = num_particle;
if dimension == 10,  num_g1 = 8;  elseif dimension == 30,  num_g1 = 16;  end     % The num_g1 is the CL population particles' number
num_g2 = num_g-num_g1;           % The num_g2 is the DMS population particles' number
group_ps = 3;                    % each sub-swarm sizes
group_num = num_g2/group_ps;     % total number of sub-swarms 

j = 0:(1/(num_g-1)):1;           % CL strategy learning probability curve Pc
j = j*10;
Pc = ones(dimension,1)*(0.05+((0.45).*(exp(j)-exp(j(1)))./(exp(j(num_g))-exp(j(1)))));      
Weight = 0.99-(1:max_iteration)*0.79/max_iteration;                                     % Linear decrease inertia weight
Weight1 = 0.99 + (0.2-0.99)*(1./(1 + exp(-5*(2*(1:max_iteration)/max_iteration - 1)))); % Nonlinear decrease inertia weight(Sigmoid function)
C = 0.15;   % Modified constants of nonlinear decrease inertia weight based on every DMS sub-swarm overall state                                                                            
            
c1 = 2.5-(1:max_iteration)*2/max_iteration;                   % Acceleration Coefficients
c2 = 0.5+(1:max_iteration)*2/max_iteration; 
pm = 0.1;                                                     % The DMS subpopulation mutation probability
flag = 0;  gap1 = 5; sigmax = 1;   sigmin = 0.1;  sig = 1;    % The whole population Guassian mutation parameter sets(全局最优值高斯变异算子参数)
% **********************Initialization*************************************
VRmin = range(1)*ones(num_g,dimension);              % Range for initial particles elements
VRmax = range(2)*ones(num_g,dimension);
interval = VRmax-VRmin;
v_max = 0.5 * interval;                              % velocity range
v_min = -v_max;
pos = VRmin+ interval.*rand(num_g,dimension);        % position
vel = v_min+(v_max-v_min).*rand(num_g,dimension);    % velocity

k=0;     fitcount=0;
result = benchmark_func(pos,func_num);               % calculation fitness function
fitcount = fitcount + num_g;
pbest_pos = pos; 
pbest_val = result';
[gbest_val,g_index] = min(result);
gbest_pos = pos(g_index,:); 
g_res(1:fitcount) = gbest_val;

obj_func_slope=zeros(num_g,1);                      % 
fri_best=(1:num_g1)'*ones(1,dimension);
        
for i = 1:num_g1                                    % Updateding examplers for the CL subpopulation(first time)
    fri_best(i,:) = i*ones(1,dimension);
    friend1 = ceil(num_g*rand(1,dimension));
    friend2 = ceil(num_g*rand(1,dimension));
    friend = (pbest_val(friend1)<pbest_val(friend2)).*friend1+(pbest_val(friend1)>=pbest_val(friend2)).*friend2;
    toss = ceil(rand(1,dimension)-Pc(:,i)');
    if toss == ones(1,dimension)
       temp_index = randperm(dimension);           
       toss(1,temp_index(1)) = 0;
       clear temp_index;
    end
    fri_best(i,:) = (1-toss).*friend+toss.*fri_best(i,:);
    for d = 1:dimension
        fri_best_pos(i,d) = pbest_pos(fri_best(i,d),d);
    end
end

for i = 1:group_num                                % Updateding examplers for the DMS population
    group_id(i,:) = [((i-1)*group_ps+num_g1+1):i*group_ps+num_g1];
    pos_group(group_id(i,:)) = i;
    [gbestval(i),gbestid] = min(pbest_val(group_id(i,:)));    % initialize the local best fitness value
    gbest(i,:) = pbest_pos(group_id(i,gbestid),:);            % initialize the local best position
end

count = 0;   index = []; m = 1; 
while k <= max_iteration && fitcount <= max_FES    % 0.9*
        if m <= max_iteration                                             % Diversity measure value 
            ava_pos = mean(pos);                                          
            Div_whole(m) = PSO_Diversity(pos,ava_pos);                    % the whole population diversity measure
            Div_CL(m) = PSO_Diversity(pos(1:num_g1,:),ava_pos);           % the CL subpopulation diversity measure
            Div_DMS(m) = PSO_Diversity(pos(num_g1+1:num_g,:),ava_pos);    % the DMS subpopulation diversity measure
            m = m + 1;    
        end
        
        k=k+1;
        Average_g = mean(result);    % the mean value of the whole population fitness
        Fit_std_g = std(result);     % the std value of the whole population fitness
        
        for i = 1:group_num
            Average_n(group_id(i,:)) = mean(result(group_id(i,:)));         % each DMS sub-swarm mean fitness value
            Fit_std_n(group_id(i,:)) = std(result(group_id(i,:)));          % each DMS sub-swarm std fitness value
        end
        % The CL subpopulation update velocity and position
        gbest_pos_temp = repmat(gbest_pos,num_g1,1);  
        delta_g1 = (c1(k).*rand(num_g1,dimension).*(fri_best_pos(1:num_g1,:)-pos(1:num_g1,:))) + (c2(k).*rand(num_g1,dimension).*(gbest_pos_temp-pos(1:num_g1,:)));
        vel_g1 = Weight(k)*vel(1:num_g1,:)+delta_g1;
        vel_g1 = ((vel_g1<v_min(1:num_g1,:)).*v_min(1:num_g1,:))+((vel_g1>v_max(1:num_g1,:)).*v_max(1:num_g1,:))+(((vel_g1<v_max(1:num_g1,:))&(vel_g1>v_min(1:num_g1,:))).*vel_g1);
        pos_g1 = pos(1:num_g1,:)+vel_g1;

        % The DMS subpopulation update velocity and position
        for i = num_g1 + 1: num_g
           % ========== nonlinear adaptive decrease inertia weight parameter============ 
           if Average_n(i) >= Average_g
               wx(i) = Weight1(k) + C;   if wx(i)>0.99,  wx(i) = 0.99;end   % 
           else     % average_n < Average_g
               wx(i) = Weight1(k) - C;   if wx(i)<0.20,  wx(i) = 0.20;end   % 
           end
           % ===================================
           delta_g2(i,:) = (c1(k).*rand(1,dimension).*(pbest_pos(i,:)-pos(i,:)))+(c2(k).*rand(1,dimension).*(gbest(pos_group(i),:)-pos(i,:)));
           vel_g2(i,:) = wx(i)*vel(i,:)+ delta_g2(i,:);    % 
           vel_g2(i,:) = ((vel_g2(i,:)<v_min(i,:)).*v_min(i,:))+((vel_g2(i,:)>v_max(i,:)).*v_max(i,:))+(((vel_g2(i,:)<v_max(i,:))&(vel_g2(i,:)>v_min(i,:))).*vel_g2(i,:));
           pos_g2(i,:) = pos(i,:) + vel_g2(i,:);
           pos_g2(i,:) = Non_uniform_mutation(pos_g2(i,:),pm,k,max_iteration,range);     % Perform the non-uniform mutation operator
        end
        
        % The whole population
        pos_g2(1:num_g1,:) = []; 
        vel_g2(1:num_g1,:) = [];   
        pos=[pos_g1;pos_g2];
        vel=[vel_g1;vel_g2];
        for i=1:num_g   
            if (sum(pos(i,:)>VRmax(i,:))+sum(pos(i,:)<VRmin(i,:))==0)     %  
               index = [index;i];
            end
        end
        if ~isempty(index)
          result(index) = benchmark_func(pos(index,:),func_num); 
          index = [];
        end
        
        % Evaluate fitness
        for i=1:num_g   
           if (sum(pos(i,:)>VRmax(i,:))+sum(pos(i,:)<VRmin(i,:))==0)     %  
              fitcount=fitcount+1;
              if fitcount>=max_FES,  break;   end
              if  result(i) < pbest_val(i)       % update pbest value and position
                pbest_pos(i,:) = pos(i,:);   
                pbest_val(i) = result(i);
                obj_func_slope(i) = 0;
              else
                obj_func_slope(i)=obj_func_slope(i)+1;
              end          
              g_res(fitcount) = gbest_val;
           end 
        end
       
        % the whole population perform the Gaussian mutation operator on gbest
        [gbestvaltmp,ind2] = min(pbest_val);        % p -> pbestval 
        gbest_postmp = pbest_pos(ind2,:);           % y -> pbest
        if gbestvaltmp < gbest_val
            gbest_pos = gbest_postmp;
            gbest_val = gbestvaltmp;
            flag = 0;
        else
            flag = flag+1;
        end
        for i = 1:num_g
            if i > num_g1  &&  pbest_val(i) < gbestval(pos_group(i))    % update local gbest value and postion
                gbest(pos_group(i),:) = pbest_pos(i,:);
                gbestval(pos_group(i)) = pbest_val(i);
            end
        end
        if flag >= gap1
            pt = gbest_pos;
            d1 = unidrnd(dimension);    randdata = 2 * rand(1,1)-1;
            pt(d1) = pt(d1)+sign(randdata)*(range(2)-range(1))*normrnd(0,sig^2);  % plus the guassian muation value
            pt(find(pt(:)>range(2))) = range(2) * rand;                           % control the boundary 
            pt(find(pt(:)<range(1))) = range(1) * rand;
            cv = benchmark_func(pt,func_num);                                     % update the optimum fitness value after the guassian muation opterator
            fitcount = fitcount+1;
            g_res(fitcount) = cv;
            if cv < gbest_val
                gbest_pos = pt;
                gbest_val = cv;
                flag=0;       
            end           
        end         
        sig = sigmax - (sigmax-sigmin)*(fitcount/max_FES);      
        
       % updating exemplar for the CL subpopulation 
       for i=1:num_g1             
            if obj_func_slope(i)>5
                fri_best(i,:)=i*ones(1,dimension);          % for its own pbest
                friend1=ceil(num_g1*rand(1,dimension));     % num_g1
                friend2=ceil(num_g1*rand(1,dimension));
                friend=(pbest_val(friend1)<pbest_val(friend2)).*friend1+(pbest_val(friend1)>=pbest_val(friend2)).*friend2;
                toss=ceil(rand(1,dimension)-Pc(:,i)');
            
                if toss==ones(1,dimension)
                    temp_index=randperm(dimension);
                    toss(1,temp_index(1))=0;
                    clear temp_index;
                end
            
                fri_best(i,:)=(1-toss).*friend+toss.*fri_best(i,:);
                for d=1:dimension
                    fri_best_pos(i,d)=pbest_pos(fri_best(i,d),d);
                end
                obj_func_slope(i)=0;
            end
       end   % updating exampler for the CL subpopulation

       % Regrouping the sub-swarm particles for the DMS subpopulation
       change_flag = 0;
       for i = 1:group_num
            flag = 1;
            for kk = 1:group_ps
                flag=flag*(sum(abs(pbest_pos(group_id(i,kk),:)-gbest(i,:))<1e-3)==dimension);
            end
            if flag == 1
                change_flag = 1;  break;
            end
        end
        if change_flag == 1 || mod(k,5) == 0          % dynamic regroup the sub-swarms R = 5
            rc = randperm(num_g2) + num_g1;           % 
            group_id=[]; gbest=[]; gbestval=[];
            for i = 1:group_num
                group_id(i,:) = rc(((i-1)*group_ps + 1):i*group_ps);
                pos_group(group_id(i,:)) = i;
                [gbestval(i),gbestid] = min(pbest_val(group_id(i,:)));
                gbest(i,:) = pbest_pos(group_id(i,gbestid),:);
            end
         end   % Regrouping the sub-swarm particles for the DMS subpopulation
        
         
         check_vel=[check_vel (sum(abs(vel'))/dimension)'];         % 
         check_vel1=[check_vel1 sum(check_vel(1:num_g1,end))/num_g1];
         check_vel2=[check_vel2 sum(check_vel(num_g1+1:end,end))/num_g2];
                 
         if fitcount>=max_FES,  break;   end   
         if (k==max_iteration) && (fitcount<max_FES)
            k=k-1;
            count = count + 1;
         end
end

% fprintf('Output\n');
check_vel1 = check_vel1./(range(2)-range(1));
check_vel2 = check_vel2./(range(2)-range(1));
position = gbest_pos;
value = gbest_val;
iteration = k;
max_FES = fitcount;
% result = result_data;
Error = value-f_bias(func_num);
gbestfit = g_res - f_bias(func_num);
end

function [newpop] = Non_uniform_mutation(pop,pm,t,T,Bound)
% Non_uniform_mutation operator；
% pm is the muation probability，pop is the matrix of population individual，t is the currents run times, T is the maximal iteration times；
% Bound is the under and up boundary of the individual 
% newpop is the new individual after the mutation operator

 b = 2;                % b is a system parameter, generally take 2~5
 [ps,D]=size(pop);     
 VRmin = Bound(1);     %*ones(ps,D);     
 VRmax = Bound(2);     %*ones(ps,D);
 
 newpop = pop;
for i = 1:ps
  for j = 1:D
    if rand() < pm
       aa = rand(1,D);
       N_mm = diag(aa); 
       if round(rand()) == 0
           %det = N_mm*(Bu - A(:,i))*(1 - t/T)^b;
           newpop(i,j) = pop(i,j) + N_mm(j,j)*(VRmax - pop(i,j))*(1 - t/T)^b;
       else  %round(rand()) == 1
           %det = N_mm*(A(:,i) - Bl)*(1 - t/T)^b;
           newpop(i,j) = pop(i,j) - N_mm(j,j)*(pop(i,j) - VRmin)*(1 - t/T)^b;
       end
    end
  end
end 
end

function D = PSO_Diversity(x, ava_Xd)       
% This program is used to compute the diversity measure value of the population
% Reference：Olorunda O , Engelbrecht A P . Measuring Exploration/Exploitation in Particle Swarms using Swarm Diversity[C], 2008.

[N,D] = size(x);             
ava_Xd = repmat(ava_Xd,N,1);    
Distance = sum([x - ava_Xd].^2,2).^(1/2);
D = mean(Distance);
end