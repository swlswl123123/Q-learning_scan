function time_rec_mean = scan_learning(D, X, Y, sector)
%scan_traditional - Description
%
% Syntax: time_rec = scan_traditional(D, X, Y, sector)
%
% Long description
Omega = 2*pi/sector;
time_slot = 100000;
N = size(D, 1);
D_rec = zeros(size(D));
DX =repmat(X',[N,1]) - repmat(X,[1,N]);% DX(i,j)的值表示Xj-Xi
DY = repmat(Y',[N,1]) - repmat(Y,[1,N]);

VX = reshape(DX,[N*N,1]);
VY = reshape(DY,[N*N,1]);
V = sqrt(VX'.^2+VY'.^2);

scan_seq = zeros(2, sector);
scan_seq(1, :) = 1:sector;
scan_seq(2, :) = mod(scan_seq(1, :) + sector / 2 - 1, sector) + 1;

scan_angle = 0 : -2*pi/sector : -2*pi + 2*pi/sector;

time_rec = [];

Q_learning_table = cell(N, 1);
% Q参数
epsilon = 0;
gamma = 0.9; % decay
alpha = 0.01;
Ne = 1;
    
for ll = 1:Ne 
    D_rec = zeros(size(D));
    % 初始化Q表
    for i = 1:N
        Q_learning_table{i} = zeros(sector, 2);
    end
    
    patten = randi([1 2], size(X));
    P1 = repmat(patten-1,[1,N]);
    P2 = repmat((patten-1)',[N,1]);
    Pa = P1 + P2;
    Pa(Pa==2)=0;
    Pa = Pa + eye(size(Pa));
    
    % 定义扫描冲突
    for t = 1:time_slot
        % 扫描一周改变扫描序列模式
        if mod(t, sector) == 0
            patten_n = randi([1 2], size(X));
            P1 = repmat(patten_n-1,[1,N]);
            P2 = repmat((patten_n-1)',[N,1]);
            P_next = P1 + P2;
            P_next(P_next==2)=0;
            P_next = P_next + eye(size(P_next));
        else
            P_next = Pa;
            patten_n = patten;
        end
    
        tri_table = zeros(1, N);
        % 选择动作
        for k = 1:N
            if rand < epsilon
                [q_max, q_max_idx] = max(Q_learning_table{k}(scan_seq(patten(k), mod(t-1, sector)+1), :));
                if length(find(Q_learning_table{k}(mod(t-1, sector)+1, :) == q_max)) > 1
                    tri_table(k) = randi([1 2]);
                else
                    tri_table(k) = q_max_idx;
                end
            else
                tri_table(k) = randi([1 2]);
            end
        end
        % 环境反馈
        theta = scan_angle(mod(t-1, sector)+1);
        P = [cos(theta),sin(theta)];
        VISION = P*[VX';VY']./V;
        VISION = acos(VISION);
        VISION(isnan(VISION))=0;
        VISION(abs(VISION)>Omega/2)=0;
        VISION(VISION>0)=1;    
        VISION = reshape(VISION,[N,N]);
    
        % 收发模式对准
        F1 = repmat(tri_table-1, [N,1]);
        F2 = repmat((tri_table-1)', [1,N]);
        F = F1 + F2;
        F(F==2)=0;
        F = F + eye(size(F));
    
        % 统计成功发现节点
        % VISION = double(VISION).*double(F).*double(D).*double(Pa);
        VISION = double(VISION).*double(F).*double(D);
    
        % 根据奖励进行学习
        % R = sum(VISION, 2);
        % for i = 1:N
        %     if R(i) == 1
        %         Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), tri_table(i)) = (1-alpha) * Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), tri_table(i)) + alpha * (-1 + gamma*max(Q_learning_table{i}(scan_seq(patten_n(i), mod(t, sector)+1), :)));  
        %     else
        %         VISION(i,:) = zeros(1,N);   
        %         Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), tri_table(i)) = (1-alpha) * Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), tri_table(i)) + alpha * (0 + gamma*max(Q_learning_table{i}(scan_seq(patten_n(i), mod(t, sector)+1), :)));
        %     end
        % end
    
        D_rec = D_rec + VISION;
        D_rec(D_rec ~= 0) = 1;
        if D_rec + eye(N) == D
            time_rec = [time_rec, t];
            break;
        end
    
        Pa = P_next;
        patten = patten_n;
    end
end
time_rec_mean = mean(time_rec);
end
    