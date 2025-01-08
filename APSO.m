function [gbest,gbestval,fitcount,suc,suc_fes]= Canonical_PSO(jingdu,func_num,fhd,Dimension,Particle_Number,Max_Gen,Max_FES,VRmin,VRmax,varargin)
%%
% 将 FuzzyPSO_func 中的结构体编程替换为常规方法。
%%
fbias=[100, 200, 300, 400, 500,...
       600, 700, 800, 900, 1000,...
       1100,1200,1300,1400,1500,...
       1600,1700,1800,1900,2000,...
       2100,2200,2300,2400,2500,...
       2600,2700,2800,2900,3000 ];
    

rand('state',sum(100*clock));
me = Max_Gen;
ps = Particle_Number;
D = Dimension;
w = 0.9;
cc = [2.0 2.0];   %acceleration constants

recorded = 0;  % 达到精度时记录相关信息
suc = 0;
suc_fes = 0;

if length(VRmin) == 1
    VRmin = repmat(VRmin,1,D);
    VRmax = repmat(VRmax,1,D);
end
lu = [VRmin; VRmax];

mv = 0.2.*(VRmax-VRmin);
Vmin = -mv;
Vmax = -Vmin;

fitcount = 0;
%% Initialize the main population and velocity

% Initialize position and velocity
pos = repmat(lu(1, :), ps, 1) + rand(ps, D) .* (repmat(lu(2, :) - lu(1, :), ps, 1));
mv = 0.10*(lu(2,:) - lu(1,:));
Vmin = repmat(-mv,ps,1);
Vmax = -Vmin;
vel = Vmin+2.*Vmax.*rand(ps,D);

% Evaluate the population
fit = (feval(fhd,pos',varargin{:})-fbias(func_num))';
fitcount = fitcount + ps;
%gen = gen + 1;

% Initialize the pbest and the pbestval
pbest = pos;
pbestval = fit;

% Initialize the gbest and the gbestval
[val,g_idx] = min(pbestval);
gbestval = val;
gbest = pbest(g_idx,:);    
gbestrep=repmat(gbest,ps,1);%update the gbest
fitcount = ps;    gen = 1;  
c1=2; c2=2; state=1;
Max_Gen = Max_FES;
while fitcount < Max_FES - 100 && gen < Max_Gen
    
    % 计算个体间距离和进化因子
    f = calculate_distances(pos, g_idx);
    f = max(0, min(1, f));
    w = 1 / (1 + 1.5 * exp(-2.6 * f));
    w = max(0.4, min(0.9, w));
    % 进行模糊控制
    pre_state = state;
    pre_c1=c1; pre_c2=c2; 
    [state, c1, c2] = fuzzy_control(f, pre_state, pre_c1, pre_c2);
    gen = gen + 1; 
    %% =================== PSO: Update position and velocity ================ 
    r1 = -1 + 2 * rand(ps, D);
    r2 = -1 + 2 * rand(ps, D);
    % r1 = 1; r2 = 1;
    % 粒子群速度更新
    vel = w * vel ...
        + c1 .* r1 .* (pbest - pos) ...
        + c2 .* r2 .* (repmat(gbest, ps, 1) - pos);
    vel = (vel>Vmax).*Vmax + (vel<=Vmax).*vel;
    vel = (vel<Vmin).*Vmin + (vel>=Vmin).*vel;


    % 粒子群位置更新
    pos = pos + vel; 
    pos = (pos>VRmax).*VRmax + (pos<=VRmax).*pos; 
    pos = (pos<VRmin).*VRmin + (pos>=VRmin).*pos;

    fit = (feval(fhd,(pos)',func_num)-fbias(func_num))';  

    % 更新 pbest 及 pbestval
    improved=(pbestval>fit);     % individuals are improved or not
    temp=repmat(improved,1,D);
    pbest=temp.*pos+(1-temp).*pbest;
    pbestval=improved.*fit+(1-improved).*pbestval;      % update the pbest
    
    % 更新 gbest 及 gbestval
    [gbestval,gbestid] = min(pbestval);  % 注意方括号不能少
    gbest = pbest(gbestid,:); %initialize the gbest and the gbest's fitness value

end

function f = calculate_distances(P,g_idx)
    num_particles = size(P, 1);
    distances = zeros(num_particles, num_particles);

    for i = 1:num_particles
        for j = 1:num_particles
            if i ~= j
                distances(i, j) = norm(P(i,:) - P(j,:));
            end
        end
    end

    mean_distances = mean(distances, 2);
    dg = mean_distances(g_idx); % 假设第一个粒子是全局最佳粒子
    dmax = max(mean_distances);
    dmin = min(mean_distances);
    f = (dg - dmin) / (dmax - dmin);
end
end
function [state, c1, c2] = fuzzy_control(f, pre_state, pre_c1, pre_c2)
    
    % 种群结构划分并计算隶属度
    if (pre_state == 1 || pre_state == 4) && (0.4 < f)&&(f < 0.7) %'Exploration';
        state =  1; 
        S1 = [0, 0, 5*f-2, 1, -10*f+8, 0];
        membership_S1 = max(0, min(1, S1));
        membership = membership_S1/5;
        c1 = 2 + 1.5*sum(membership);
        c2 = 2 - 1.5*sum(membership);
        
    elseif  (pre_state == 2 || pre_state == 1) && (0.2 < f)&&(f < 0.6) %'Exploitation';
        state =  2;     
        S2 = [0, 10*f-2, 1, -5*f+3, 0];
        membership_S2 = max(0, min(1, S2));
        membership = membership_S2/5;
        c1 = 2 + 0.5*sum(membership);
        c2 = 2;

    elseif  (pre_state == 3 || pre_state == 2) && (0.3 >= f) %'Convergence';
        state =  3;
        S3 = [1, -5*f+1.5, 0];
        membership_S3 = max(0, min(1, S3)); 
        membership = membership_S3/3;
        c1 = 2 + 0.5*sum(membership);
        c2 = 2 + 0.5*sum(membership);
  
    elseif (pre_state == 4 || pre_state == 3) && (0.7 <= f) %'Jumping Out';
        state =  4;  
        S4 = [0, 5*f-3.5, 1];
        membership_S4 = max(0, min(1, S4));
        membership = membership_S4/3;
        c1 = 2 - 2*sum(membership);
        c2 = 2 + 2*sum(membership);

    else
        state = pre_state;
        c1 = pre_c1;
        c2 = pre_c2;
    end

end



