function time_rec_mean = scan_learn_single(D, X, Y, sector)
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
    epsilon = -1;
    gamma = 0.9; % decay
    alpha = 0.01;
    Ne = 1;
        
    for ll = 1:Ne 
        r_rec = [];
        r_all = 0;  
        D_rec = zeros(size(D));
        % 初始化Q表
        for i = 1:N
            Q_learning_table{i} = zeros(sector, 2);
        end
    
        % 定义扫描冲突
        for t = 1:time_slot
            if mod(t, sector) == 0
                r_rec = [r_rec, r_all];
                r_all = 0;
            end
            % 扫描一周改变扫描序列模式
            tri_table = randi([1 2], [1, N]);
            ptr_table = zeros(1, N);
            % 选择动作
            for k = 1:N
                if rand < epsilon
                    [q_max, q_max_idx] = max(Q_learning_table{k}(mod(t-1, sector)+1, :));
                    if length(find(Q_learning_table{k}(mod(t-1, sector)+1, :) == q_max)) > 1
                        ptr_table(k) = randi([1 2]);
                    else
                        ptr_table(k) = q_max_idx;
                    end
                else
                    ptr_table(k) = randi([1 2]);
                end
            end
            ptr = (ptr_table-1)*2-1;
            ptr = repmat(ptr, [N, 1]);
            ptr = reshape(ptr, [1 N*N]);
            P1 = repmat(ptr_table-1, [N,1]);
            P2 = repmat((ptr_table-1)', [1,N]);
            Pt = P1 + P2;
            Pt(Pt==2)=0;
            Pt = Pt + eye(size(Pt));
            % 收发模式对准
            F1 = repmat(tri_table-1, [N,1]);
            F2 = repmat((tri_table-1)', [1,N]);
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
    
            % 统计成功发现节点
            % VISION = double(VISION).*double(F).*double(D).*double(Pa);
        
            % 根据奖励进行学习
            R = sum(VISION, 2);
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
    
                if R(i) > 1
                    r = 1;
                    r_all  = r_all + 1;
                else
                    r = 0;
                end
    
                Q_learning_table{i}(mod(t-1, sector)+1, ptr_table(i)) = (1-alpha) * Q_learning_table{i}(mod(t-1, sector)+1, ptr_table(i)) + alpha * (r + gamma*max(Q_learning_table{i}(mod(t, sector)+1, :)));
            end
    
            template = ones(N,N);
            for i = 1:N
                if R(i) > 1
                    template(i, :) = zeros(1, N);
                    template(:, i) = zeros(N, 1);
                end
            end
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
        