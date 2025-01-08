clear all
clc

global fbias
warning('off')

D=10;
Xmin=-100;
Xmax=100;

pop_size=4*D;%2^(4+floor(log2(sqrt(D))));
fes_max=1000*D;
iter_max=ceil(fes_max/pop_size);

runtimes=3;
fhd=str2func('cec17_func');
fbias=[100, 200, 300, 400, 500,...
       600, 700, 800, 900, 1000,...
       1100,1200,1300,1400,1500,...
       1600,1700,1800,1900,2000,...
       2100,2200,2300,2400,2500,...
       2600,2700,2800,2900,3000 ];

jingdu=0;

funset=[1:30];

for fun=1:length(funset)
    func_num=funset(fun);
    suc_times = 0; 
    
    fesusage=0;
    count=0;
    
    for runs=1:runtimes
        suc = 0;
        suc_fes = 0;   % added by us
    
%         pop_size=2*D;iter_max=ceil(fes_max/pop_size);
        [gbest,gbestval,FES]= APSO(jingdu,func_num,fhd,D,pop_size,iter_max,fes_max,Xmin,Xmax,func_num);

        t=toc;
        time_usage(runs,fun)=t;
        
        xbest(runs,:)=gbest;
        fbest(runs,func_num)=gbestval;
        fprintf('�� %d �����е����Ž��Ϊ��%1.4e\n',runs,gbestval);
        suc_times = suc_times + suc;
        if suc == 1  % ���ﵽ�趨����ʱ��ͳ�����ʱ           
            fesusage = fesusage + suc_fes;   % �ﵽ����ʱ�� fes            
        end        
    end
      
    %% ����Ϊ�����������趨��⾫�������µĳɹ��ʡ��������۴����Լ�����ʱ��
    SR(1,func_num) = suc_times/runtimes;
    if suc_times>0
        FEs(1,func_num) = fesusage/suc_times;  % ���㾫�ȵĶ�����������ĵ�ƽ�� fes��δ���ǲ����㾫�ȵ�����
        SP (1,func_num) = fes_max*(1-SR(1,func_num))/SR(1,func_num) + FEs(1,func_num); % �ۺ��������㷨�����ܣ��ȿ��ǳɹ��ģ�Ҳ����δ�ɹ���
%         tu(1,func_num) = timeusage/suc_times;
    else
        FEs(1,func_num) = -1;  % ���㾫�ȵĶ�����������ĵ�ƽ�� fes��δ���ǲ����㾫�ȵ�����
        SP (1,func_num) = -1; % �ۺ��������㷨�����ܣ��ȿ��ǳɹ��ģ�Ҳ����δ�ɹ���
%         tu(1,func_num) = -1;
    end
    
    f_mean = mean(fbest(:,func_num));
    f_std  = std(fbest(:,func_num));
    f_SR   = SR(1,func_num);

    fprintf('\nFunction F%d :\nAvg. fitness = %1.3e(%1.3e)\n\n',func_num, f_mean, f_std);    
    fprintf(' -------------------------------------------------- \n');
    %% �������ݵ� Excel
    % filename = 'APSO_CEC2017.xlsx'; 
    % data_to_save = [func_num, f_mean, f_std, f_SR];
    % header = {'Func Num', 'Mean', 'Std Dev', 'Time'};
    % 
    % if func_num == 1
    %     % �ڵ�һ��д��ʱ��д����������
    %     xlswrite(filename, header, 'Sheet1', 'A1');
    %     xlswrite(filename, data_to_save, 'Sheet1', 'A2');
    % else
    %     % ׷���������ܵ�����
    %     xlswrite(filename, data_to_save, 'Sheet1', ['A' num2str(func_num + 1)]);
    % end
    
end



