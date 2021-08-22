function time_rec_mean = scan_traditional(D, X, Y, sector)
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

% 天线扫描角度
scan_angle = 0 : -2*pi/sector : -2*pi + 2*pi/sector;

% 记录结果
time_rec = [];
Ne = 10;
for ll = 1:Ne 

    D_rec = zeros(size(D));
    tri_table = randi([1 2], size(X));
    tri = (tri_table-1)*2-1;
    tri = repmat(tri, [N, 1]);
    tri = reshape(tri, [1 N*N]);
    % 定义扫描冲突
    for t = 1:time_slot
        % 扫描一周改变扫描序列模式
        if mod(t, sector) == 1
            tri_table = randi([1 2], size(X));
            tri = (tri_table-1)*2-1;
            tri = repmat(tri, [N, 1]);
            tri = reshape(tri, [1 N*N]);
        end
        % 收发模式对准
        F1 = repmat((tri_table-1)', [N,1]);
        F2 = repmat((tri_table-1), [1,N]);
        F = F1 + F2;
        F(F==2)=0;
        F = F + eye(size(F));

        % 环境反馈
        theta = scan_angle(mod(t-1, sector)+1);
        P = [cos(theta),sin(theta)];
        VISION = P*[VX';VY'].*tri./V;
        VISION = reshape(VISION,[N,N])';
        % for i = 1:N
        %     for j = 1:i
        %         VISION(i, j) = -VISION(i, j);
        %     end
        % end
        VISION = acos(VISION);
        VISION = VISION .* double(F);
        VISION(isnan(VISION))=0;
        VISION(abs(VISION)>Omega/2)=0;
        VISION(VISION>0)=1;    


        % 统计成功发现节点
        % VISION = double(VISION).*double(F);
        % VISION = double(F).*double(D);
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
                    isconfilct(i) = 1;
                    isconfilct(node_recv_conflict) = 1;
                end
            else
                if R(i) > 1
                    template(i, :) = zeros(1, N);
                    template(:, i) = zeros(N, 1);
                    isconfilct(i) = 1;
                    isconfilct(find(VISION(i, :) == 1)) = 1;
                end
            end
        end
        % isconfilct;
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
