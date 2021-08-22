clear all; close all;

% 参数设置
N = 16;
sector = 8;
Omega = 2*pi/sector;
time_slot = 100000;
HOP = 3;
W = 2;
H = 2;
Ne = 1;

% 生成拓扑
[X,Y,D,f1] = mknet(N,HOP,W,H);
% DX =repmat(X',[N,1]) - repmat(X,[1,N]);% DX(i,j)的值表示Xj-Xi
% DY = repmat(Y',[N,1]) - repmat(Y,[1,N]);
% Dist = sqrt(DX.^2+DY.^2);

% VX = reshape(DX,[N*N,1]);
% VY = reshape(DY,[N*N,1]);
% V = sqrt(VX'.^2+VY'.^2);
% 统计各个波束内平均几个点

san_seq = zeros(2, sector);
san_seq(1, :) = 0 : -2*pi/sector : -2*pi + 2*pi/sector;
san_seq(2, :) = mod(san_seq(1, :) + pi, 2*pi);

res = [];
for ll = 1 : Ne

D_rec = zeros(size(D));
tri_table = randi([1, 2], [1, N]);

% 定义扫描冲突
for t = 1:time_slot
    DD = zeros(size(D));
    for i = 1:N
        theta = san_seq(tri_table(i), mod(t-1, sector)+1); % 当前节点扫描方向
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
        candidate = [];
        for j = 1:length(scan_area)
            nid = scan_area(j);
            if tri_table(i) == tri_table(nid)
                continue;
            end
            % 天线已对准的节点
            % 判断是否唯一
            dx = X - X(nid);
            dy = Y - Y(nid);
            theta_nid = san_seq(tri_table(nid), mod(t-1, sector)+1);
            theta_cover = [cos(theta_nid), sin(theta_nid)]*[dx';dy'] ./ sqrt(dx'.^2 + dy'.^2);
            theta_cover = acos(theta_cover);
            theta_cover(isnan(theta_cover)) = 0;
            theta_cover(theta_cover>Omega/2) = 0;
            theta_cover(theta_cover>0) = 1;
            theta_cover(find(tri_table == tri_table(nid))) = 0;
            theta_cover = theta_cover .* double(D(nid, :));
            if length(find(theta_cover == 1)) == 1
                candidate = [candidate; [i, nid]];
            end
        end

        if size(candidate, 1) == 1
            DD(candidate(1), candidate(2)) = 1;
            DD(candidate(2), candidate(1)) = 1;
        else
            disp('conflict');
        end

        % 判断是否冲突
        % if cnt_conflict ~= 1
        %     if cnt_conflict > 1
        %         % disp(cnt_conflict)                
        %     end
        %     DD(i, :) = D_tmp;
        % else
        %     DD(find(DD(i, :) == 1), i) = 1;
        % end
    end 
    D_rec = D_rec + DD;
    D_rec(D_rec ~= 0) = 1;
    if D_rec + eye(N) == D
        % disp(t);
        res = [res, t];
        break;
    end
    % 扫描一周改变收发模式
    if mod(t, sector) == 0
        tri_table = randi([1, 2], [1, N]);
    end
end
end
mean(res)