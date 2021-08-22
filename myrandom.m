function time_rec_mean = myrandom(D, X, Y, sector)
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
    epsilon = 0;
    gamma = 0.3; % decay 0.3
    alpha = 0.05;
    % Q参数
    Ne = 1;
        
    for ll = 1:Ne 
        D_rec = zeros(size(D));
        tri_table = randi([1 2], [1, N]);
        for i = 1:N
            Q_learning_table{i} = zeros(2, sector);
        end
        % 定义扫描冲突
        for t = 1:time_slot
            % 扫描一周改变扫描序列模式
            % 选择动作
            for k = 1:N
                if rand < epsilon
                    [q_max, q_max_idx] = max(Q_learning_table{k}(tri_table(k), :));
                    if length(find(Q_learning_table{k}(tri_table(k), :) == q_max)) > 1
                        ptr_table(k) = randi([1 sector]);
                    else
                        ptr_table(k) = q_max_idx;
                    end
                else
                    ptr_table(k) = randi([1 sector]);
                end
            end
            theta = scan_angle(ptr_table);
            theta = repmat(theta, [N 1]);
            theta = reshape(theta, [1, N*N]);
            % ptr = (ptr_table-1)*2-1;
            % ptr = repmat(ptr, [N, 1]);
            % ptr = reshape(ptr, [1 N*N]);
            % P1 = repmat(ptr_table-1, [N,1]);
            % P2 = repmat((ptr_table-1)', [1,N]);
            % Pt = P1 + P2;
            % Pt(Pt==2)=0;
            % Pt = Pt + eye(size(Pt));
            % 收发模式对准
            F1 = repmat(tri_table-1, [N,1]);
            F2 = repmat((tri_table-1)', [1,N]);
            F = F1 + F2;
            F(F==2)=0;
            F = F + eye(size(F));
            % 环境反馈
            % theta = scan_angle(mod(t-1, sector)+1);
            P = [cos(theta);sin(theta)];
            VISION = (P(1,:).*VX' + P(2,:).*VY')./V;
            VISION = reshape(VISION,[N,N])';
            VISION = acos(VISION);
            for i = 1:N
                for j = i+1:N
                    if VISION(i, j) ~= VISION(j ,i)
                        VISION(i, j) = 0;
                        VISION(j, i) = 0;
                    end
                end
            end
            VISION = VISION .* double(F);
            VISION(isnan(VISION))=0;
            VISION(abs(VISION)>Omega/2)=0;
            VISION(VISION>0)=1;    
    
            % 统计成功发现节点
            % VISION = double(VISION).*double(F).*double(D).*double(Pa);
        
            % 根据奖励进行学习
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
                        if R(node_recv(1)) == 1
                            template(i, :) = zeros(1, N);
                            isconfilct(i) = 1;
                        else
                            template(i, :) = zeros(1, N);
                            template(:, i) = zeros(N, 1);
                        end
                        % if R(node_recv(1)) == 1
                        %     isconfilct(i) = 2;
                        % else
                        %     isconfilct(i) = 1;
                        % end
                        % isconfilct(node_recv_conflict) = 1;
                    end
                else
                    if R(i) > 1
                        template(i, :) = zeros(1, N);
                        template(:, i) = zeros(N, 1);
                        isconfilct(i) = 1;
                        % isconfilct(find(VISION(i, :) == 1)) = 1;
                    end       
                end
            end

            % r = 0;
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
    
                % if R(i) > 1
                %     r = 1;
                % else
                %     r = 0;
                % end
                state = randi([1 2]);
                Q_learning_table{i}(tri_table(i), ptr_table(i)) = (1-alpha) * Q_learning_table{i}(tri_table(i), ptr_table(i)) + alpha * (isconfilct(i) + gamma*max(Q_learning_table{i}(state, :)));
                tri_table(i) = state;
            end
    
            % template = ones(N,N);
            % for i = 1:N
            %     if R(i) > 1
            %         template(i, :) = zeros(1, N);
            %         % template(:, i) = zeros(N, 1);
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
    end
    time_rec_mean = mean(time_rec);
    end
        