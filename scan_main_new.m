clear all; close all;

% 参数设置
N = 50;
sector = 8;
Omega = 2*pi/sector;
time_slot = 100000;
HOP = 3;
W = 2;
H = 2;

% 生成拓扑
[X,Y,D] = mknet(N,HOP,W,H);
D_rec = zeros(size(D));
DX =repmat(X',[N,1]) - repmat(X,[1,N]);% DX(i,j)的值表示Xj-Xi
DY = repmat(Y',[N,1]) - repmat(Y,[1,N]);
Dist = sqrt(DX.^2+DY.^2);

VX = reshape(DX,[N*N,1]);
VY = reshape(DY,[N*N,1]);
V = sqrt(VX'.^2+VY'.^2);

% 天线扫描角度
scan_angle = 0 : -2*pi/sector : -2*pi + 2*pi/sector;

% 记录结果
time_rec = [];
Ne = 1;

for ll = 1:Ne 

D_rec = zeros(size(D));
tri_table = randi([1 2], size(X));

% 定义扫描冲突
for t = 1:time_slot
    % 扫描一周改变扫描序列模式
    if mod(t, sector) == 1
        tri_table = randi([1 2], size(X));
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
    F1 = repmat((tri_table-1)', [N,1]);
    F2 = repmat((tri_table-1), [1,N]);
    F = F1 + F2;
    F(F==2)=0;
    F = F + eye(size(F));

    % 统计成功发现节点
    VISION = double(VISION).*double(F).*double(D);
    
    % 判断是否冲突
    R = sum(VISION, 2);
    for i = 1:N
        if R(i) > 1
            VISION(i, :) = zeros(1, N);
        end
    end

    D_rec = D_rec + VISION;
    D_rec(D_rec ~= 0) = 1;
    if D_rec + eye(N) == D
        time_rec = [time_rec, t];
        break;
    end
end
end
mean(time_rec)