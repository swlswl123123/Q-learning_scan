function time_rec_mean = scan_traditional_cmp(D, X, Y, sector)
%scan_traditional - Description
%
% Syntax: time_rec = scan_traditional(D, X, Y, sector)
%
% Long description
Omega = 2*pi/sector;
time_slot = 1000000;
N = size(D, 1);
D_rec = zeros(size(D));
DX =repmat(X',[N,1]) - repmat(X,[1,N]);% DX(i,j)的值表示Xj-Xi
DY = repmat(Y',[N,1]) - repmat(Y,[1,N]);
Dist = sqrt(DX.^2+DY.^2);

VX = reshape(DX',[N*N,1]);
VY = reshape(DY',[N*N,1]);
V = sqrt(VX'.^2+VY'.^2);

% 天线扫描角度
scan_angle = 0 : -2*pi/sector : -2*pi + 2*pi/sector;

% 记录结果
time_rec = [];
Ne = 1;
for ll = 1:Ne 

    D_rec = zeros(size(D));
    tri_table = zeros(size(X));
    tri = tri_table;
    % 定义扫描冲突
    for t = 1:time_slot
        % 选择收发模式，收发与指向绑定
        tri_table = randi([1 2], size(X));
        tri = (tri_table-1)*2-1;
        tri = repmat(tri, [N, 1]);
        tri = reshape(tri, [1 N*N]);
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
        VISION = acos(VISION);
        VISION = VISION .* double(F);
        VISION(isnan(VISION))=0;
        VISION(abs(VISION)>Omega/2)=0;
        VISION(VISION>0)=1;    
        
        % 判断是否冲突
        % R = sum(VISION, 2);
        % for i = 1:N
        %     if R(i) > 1
        %         VISION(i, :) = zeros(1, N);
        %         VISION(:, i) = zeros(N, 1);
        %     end
        % end
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
