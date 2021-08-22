clear all; close all;

% 参数设置
N = 48;
sector = 8;
Omega = 2*pi/sector;
time_slot = 10000;
HOP = 6;
W = 3;
H = 3;
% reward
a = 0.1;
b = 0.05;

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
san_seq(1, :) = 0 : -pi/4 : -2*pi + pi/4;
% san_seq(2, :) = pi : -pi/4 : -pi + pi/4;
probablity = ones(N, sector) * (1 / sector);

tri_table = randi([1, 2], [1, N]);

% 定义扫描冲突
for t = 1:time_slot
    tri_table = randi([1, 2], [1, N]);
    DD = zeros(size(D));
    seq_num = zeros(1, N);
    % 选择动作
    for k = 1:N
        seq_num(k) = ceil(rand * sector);
    end
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
            theta_nid = san_seq(seq_num(nid));
            theta_cover = (cos(theta_nid)*dx + sin(theta_nid)*dy) / sqrt(dx^2 + dy^2);
            if abs(acos(theta_cover)) <= Omega/2
                cnt_conflict = cnt_conflict + 1;
                DD(i, nid) = 1;
            end
        end
        % 判断是否冲突
        r = 0;
        if cnt_conflict ~= 1
            if cnt_conflict > 1
                % disp(cnt_conflict)                
            end
            DD(i, :) = D_tmp;
        else
            r = 1;
            DD(find(DD(i, :) == 1), i) = 1;
        end
        % 根据奖励学习
        if r == 1
            p_change = probablity(i, seq_num(i));
            probablity(i, :) = b / (sector - 1) + (1 - b) * probablity(i, :);
            probablity(i, seq_num(i)) = (1 - b) * p_change;
        else
            p_change = probablity(i, seq_num(i));
            probablity(i, :) = (1 - a) * probablity(i, :);
            probablity(i, seq_num(i)) = p_change + a * (1 - p_change);
        end
    end 
    D_rec = D_rec + DD;
    D_rec(D_rec ~= 0) = 1;
    if D_rec + eye(N) == D
        disp(t);
        break;
    end
    % 扫描一周改变收发模式
    % if mod(t, sector) == 0
    %     tri_table = randi([1, 2], [1, N]);
    % end
end