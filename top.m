clear all; close all;

sector = 12;
R = 1;
ST_mean = [];
SL_mean = [];
SLL_mean = [];
for N = 15

    Ne = 10;
    ST = [];
    SL = [];
    SLL = [];
    st = 0;
    sl = 0;
    sll = 0;
    for ll = 1:Ne
        % 生成拓扑
        [X,Y,D] = mknet(N,R);
        st = LearnCombine(D, X, Y, sector);
        sl = LearnCombine_UCB(D, X, Y, sector);
        % sll = LearnSingle(D, X, Y, sector);
        if isnan(st)
            disp('NaN')
            st = 100000;
        end
        if isnan(sl)
            disp('NaN')
            sl = 100000;
        end
        ST = [ST, st];
        SL = [SL, sl];
        SLL = [SLL, sll];
    end

    ST_mean = [ST_mean, mean(ST)]
    SL_mean = [SL_mean, mean(SL)]
    SLL_mean = [SLL_mean, mean(SLL)]

end

% csvwrite('ST_mean.csv', ST_mean);
% csvwrite('SL_mean.csv', SL_mean);