function time_rec_mean = LearnCombine(D, X, Y, sector)
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

VX = reshape(DX',[N*N,1]);
VY = reshape(DY',[N*N,1]);
V = sqrt(VX'.^2+VY'.^2);

scan_angle = 0 : -2*pi/sector : -2*pi + 2*pi/sector;

time_rec = [];
Q_learning_table = cell(N, 1);
% Q参数
epsilon = 0.3;
gamma = 0.3; % decay 0.3
alpha = 0.05;
Ne = 10;
    
for ll = 1:Ne 
    r_rec = [];
    r_all = 0;  
    D_rec = zeros(size(D));
    % 初始化Q表
    for i = 1:N
        Q_learning_table{i} = zeros(sector, 4);
    end

    % 定义扫描冲突
    for t = 1:time_slot
        if mod(t, sector) == 0
            r_rec = [r_rec, r_all];
            r_all = 0;
        end
        % 扫描一周改变扫描序列模式
        action = zeros(1, N);
        % 选择动作
        for k = 1:N
            if rand < epsilon
                [q_max, q_max_idx] = max(Q_learning_table{k}(mod(t-1, sector)+1, :));
                if length(find(Q_learning_table{k}(mod(t-1, sector)+1, :) == q_max)) > 1
                    action(k) = randi([1 4]);
                else
                    action(k) = q_max_idx;
                end
            else
                action(k) = randi([1 4]);
            end
        end
        % 00 1南发 01 2南收 10 3北发 11 4北收
        ptr_table = floor(action ./ 3);
        ptr = ptr_table*2-1;
        ptr = repmat(ptr, [N, 1]);
        ptr = reshape(ptr, [1 N*N]);
        P1 = repmat(ptr_table, [N,1]);
        P2 = repmat((ptr_table)', [1,N]);
        Pt = P1 + P2;
        Pt(Pt==2)=0;
        Pt = Pt + eye(size(Pt));
        % 收发模式对准
        tri_table = mod(action, 2);
        F1 = repmat(tri_table, [N,1]);
        F2 = repmat((tri_table)', [1,N]);
        F = F1 + F2;
        F(F==2)=0;
        F = F + eye(size(F));
        % 环境反馈
        theta = scan_angle(mod(t-1, sector)+1);
        P = [cos(theta),sin(theta)];
        VISION = P*[VX';VY'].*ptr./V;
        VISION = reshape(VISION,[N,N])';
        VISION = acos(VISION);
        VISION = VISION .* double(F).* double(Pt);
        VISION(isnan(VISION))=0;
        VISION(abs(VISION)>Omega/2)=0;
        VISION(VISION>0)=1; 
        
        % 1是发 0是收
        % 判断是否冲突
        isconfilct = zeros(1, N);
        R = sum(VISION, 2);
        template = ones(N,N);
        for i = 1:N
            if tri_table(i) == 1
                node_recv = find(VISION(i, :) == 1);
                conflict_num = length(node_recv);
                node_recv_conflict = [];
                for n = node_recv
                    if R(n) == 1 && D_rec(i, n) == 1
                        conflict_num = conflict_num - 1;
                    else
                        node_recv_conflict = [node_recv_conflict, n];
                    end
                end
                if conflict_num > 1
                    template(i, :) = zeros(1, N);
                    template(:, i) = zeros(N, 1);
                    % if R(node_recv(1)) == 1
                    %     isconfilct(i) = 2;
                    % else
                    %     isconfilct(i) = 1;
                    % end
                    isconfilct(i) = 1;
                    isconfilct(node_recv_conflict) = 1;
                end
            else
                node_send = find(VISION(i, :) == 1);
                if R(i) > 1
                    template(i, :) = zeros(1, N);
                    template(:, i) = zeros(N, 1);
                    if R(node_send(1)) == 1
                        isconfilct(i) = 2;
                    else
                        isconfilct(i) = 1;
                    end
                    % isconfilct(i) = 1;
                    isconfilct(find(VISION(i, :) == 1)) = 1;
                end
            end
        end

        % 统计成功发现节点
        % VISION = double(VISION).*double(F).*double(D).*double(Pa);
    
        % 根据奖励进行学习
        % R = zeros(1, N);
        % for i = 1:N
        %     id_idx = find(VISION(i,:) == 1);
        %     num = length(id_idx);
        %     if R(i) < num
        %         R(i) = num;
        %     end
        %     for d = id_idx
        %         if d > i
        %             R(d) = R(d) + num;
        %         end
        %     end
        % end
        r = 0;
        for i = 1:N
            % if R(i) == 1
            %     Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), ptr_table(i)) = (1-alpha) * Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), ptr_table(i)) + alpha * (-1 + gamma*max(Q_learning_table{i}(scan_seq(patten_n(i), mod(t, sector)+1), :)));  
            % else
            %     VISION(i,:) = zeros(1,N);   
            %     Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), ptr_table(i)) = (1-alpha) * Q_learning_table{i}(scan_seq(patten(i), mod(t-1, sector)+1), ptr_table(i)) + alpha * (0 + gamma*max(Q_learning_table{i}(scan_seq(patten_n(i), mod(t, sector)+1), :)));
            % end
            % if R(i) ~= 1
            %     VISION(i,:) = zeros(1,N);
            % end

            if tri_table(i) == 1
                % if isconfilct(i) == 2
                %     r = 1;
                % else
                %     r = -0.5*isconfilct(i);
                % end

                r = -isconfilct(i);
                % r_all  = r_all + 1;
            else
                if isconfilct(i) == 2
                    r = 1;
                else
                    r = -0.5*isconfilct(i);
                    % r = 0;
                end
                % r = isconfilct(i);
            end

            Q_learning_table{i}(mod(t-1, sector)+1, action(i)) = (1-alpha) * Q_learning_table{i}(mod(t-1, sector)+1, action(i)) + alpha * (r + gamma*max(Q_learning_table{i}(mod(t, sector)+1, :)));
        end

        % template = ones(N,N);
        % for i = 1:N
        %     if R(i) > 1
        %         template(i, :) = zeros(1, N);
        %         template(:, i) = zeros(N, 1);
        %     end
        % end
        VISION = VISION .* template;
        VISION(VISION ~= 0) = 1;
        D_rec = D_rec + VISION;
        D_rec(D_rec ~= 0) = 1;
        if D_rec + eye(N) == D
            time_rec = [time_rec, t];
            break;
        end
    end
    % plot(r_rec)
end
time_rec_mean = mean(time_rec);
end