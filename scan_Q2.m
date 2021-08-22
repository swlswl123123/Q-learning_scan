clear all; close all;

% 参数设置
N = 48;
sector = 8;
Omega = 2*pi/sector;
time_slot = 10000;
HOP = 6;
W = 3;
H = 3;

% 生成拓扑
[X,Y,D,f1] = mknet(N,HOP,W,H);
D_rec = zeros(size(D));
% DX =repmat(X',[N,1]) - repmat(X,[1,N]);% DX(i,j)的值表示Xj-Xi
% DY = repmat(Y',[N,1]) - repmat(Y,[1,N]);
% Dist = sqrt(DX.^2+DY.^2);

% VX = reshape(DX,[N*N,1]);
% VY = reshape(DY,[N*N,1]);
% V = sqrt(VX'.^2+VY'.^2);

san_seq = zeros(1, sector);
san_seq = 0 : -pi/4 : -2*pi + pi/4;

tri_table = randi([1, 2], [1, N]);
Q_learning_table = cell(N, 1);
% Q参数
epsilon = 0.9;
gamma = 0.8; % decay
lr = 0.01;

% 已经建立连接的节点不再发送回复造成冲突
% 初始化Q表
for i = 1:N
    Q_learning_table{i} = zeros(2, sector);
end

% 定义扫描冲突
tri_table = randi([1, 2], [1, N]);
for t = 1:time_slot
    DD = zeros(size(D));
    seq_num = zeros(1, N);
    % 扫描一周改变收发模式
    if t ~= 1
        tri_table = tri_table_next;
    end
    % 选择动作
    for k = 1:N
        if rand < epsilon
            [q_min, q_min_idx] = min(Q_learning_table{k}(tri_table(k), :));
            if length(find(Q_learning_table{k}(tri_table(k), :) == q_min)) > 1
                tmp = find(Q_learning_table{k}(tri_table(k), :) == q_min);
                tmp_idx = randi([1 length(tmp)]);
                seq_num(k) = tmp(tmp_idx);
            else
                seq_num(k) = q_min_idx;
            end
        else
            seq_num(k) = randi([1 sector]);
        end
    end

    % if mod(t, sector) == 0
    %     tri_table_next = randi([1, 2], [1, N]);
    % else
    %     tri_table_next = tri_table;
    % end
    tri_table_next = randi([1, 2], [1, N]);

    for i = 1:N
        theta = san_seq(seq_num(i)); % 当前节点扫描方向
        P = [cos(theta), sin(theta)];
        VX = X - X(i);
        VY = Y - Y(i);
        V = sqrt(VX'.^2 + VY'.^2);
        VISION = P*[VX';VY']./V;
        VISION = acos(VISION);
        VISION(isnan(VISION))=0;
        VISION(abs(VISION)>Omega/2)=0;
        VISION(VISION>0)=1;
        VISION = VISION .* double(D(i, :));
        % vision 矩阵代表在该方向上的节点
        % 依次判断所在区域内节点是否与该节点冲突
        % 该节点为发送模式，则有2个以上接收节点指向此节点为冲突
        % 该节点为接收模式，则有2个以上发送节点指向此节点为冲突
        cnt_conflict = 0;
        D_tmp = DD(i, :);
        scan_area = find(VISION ~= 0);
        for j = 1:length(scan_area)
            nid = scan_area(j);
            if tri_table(i) == tri_table(nid)
                continue;
            end

            dx = X(i) - X(nid);
            dy = Y(i) - Y(nid);
            theta_nid = san_seq(mod(t-1, sector)+1);
            theta_cover = (cos(theta_nid)*dx + sin(theta_nid)*dy) / sqrt(dx^2 + dy^2);
            % 已发现便不再回复造成干扰
            if abs(acos(theta_cover)) <= Omega/2 && ((tri_table(i) == 1 && DD(i, nid) == 0) || tri_table(i) == 0)
            % if abs(acos(theta_cover)) <= Omega/2
                cnt_conflict = cnt_conflict + 1;
                DD(i, nid) = 1;
            end
        end
        % 判断是否冲突
        r = 0;
        if cnt_conflict ~= 1
            if cnt_conflict > 1
                % disp(cnt_conflict)
            else
                r = 1;                
            end
            DD(i, :) = D_tmp;
        else
            r = 1;
            DD(find(DD(i, :) == 1), i) = 1;
        end
        % if sum(sum(abs(DD' - DD))) ~= 0
        %     disp('wrong')
        %     return;
        % end
        % 根据奖励学习
        cur_state = tri_table(i);
        next_state = tri_table_next(i);
        q_predict = Q_learning_table{i}(cur_state, seq_num(i));
        q_target = r + gamma * min(Q_learning_table{i}(next_state, :));
        Q_learning_table{i}(cur_state, seq_num(i)) = Q_learning_table{i}(cur_state, seq_num(i)) + lr*(q_target - q_predict);
    end 
    D_rec = D_rec + DD;
    % if sum(sum(abs(D_rec' - D_rec))) ~= 0
    %     disp('Dwrong')
    %     return;
    % end
    D_rec(D_rec ~= 0) = 1;
    if D_rec + eye(N) == D
        disp(t);
        break;
    end
end