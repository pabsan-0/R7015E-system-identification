format compact
format short
set(gca,'FontName','Times');
%%  SYNTHETIC SYSTEM 1

% propagation and measurement equation matrices
A = [1 1; 0 1];
B = [0; 1];
C = [1 0.5];
Q = [20 0; 0 0.1];  % covariance matrix propagation error
R = [4];            % covariance matrix measurement error         
LL = eye(2);        % matrix coefficient for w

% checking system's observability & controlability
W = ctrb(A, B); rank(W); 
V = obsv(A, C); rank(V); 

LuenbergObserverPoles = [0.01 0.02];


%% DS1
filename = ['Dataset1.txt']
[u, y] = pablo_load_data(filename);
yhat_luenberg = pablo_luenberg(u, y, LuenbergObserverPoles, A, B, C);
[yhat_kalman_tv, L_stat, P_stat] = pablo_kalman_tv(u, y, A, B, C, LL, Q, R);
yhat_kalman_ti = pablo_kalman_ti(u, y, L_stat, A, B, C);

a = (yhat_luenberg-y')';
b = (yhat_kalman_tv-y')';
c = (yhat_kalman_ti-y')';

figure(1); clf; sgtitle(filename); 
subplot(3,2,1); plot(1:1000, a, 'r'); yticks([-50 0 50]); ylabel('Luenberg') 
subplot(3,2,3); plot(1:1000, b, 'g'); yticks([-50 0 50]); ylabel('TV Kalman') 
subplot(3,2,5); plot(1:1000, c, 'b'); yticks([-50 0 50]); ylabel('TI Kalman') 
subplot(3,2,[2 4 6]); boxplot([a,b,c],'Colors',['r' 'g' 'b'],'Labels',{'L','TI K', 'TV K'}); 
yticks([-120 -100 -80 -60 -40 -20 0 20 40 60 80 100 120])

MSE1 = [immse(yhat_luenberg, y') ...
       immse(yhat_kalman_tv, y') ...
       immse(yhat_kalman_ti, y')]
   
%% DS2
filename = ['Dataset2.txt']
[u, y] = pablo_load_data(filename);
yhat_luenberg = pablo_luenberg(u, y, LuenbergObserverPoles, A, B, C);
[yhat_kalman_tv, L_stat, P_stat] = pablo_kalman_tv(u, y, A, B, C, LL, Q, R);
yhat_kalman_ti = pablo_kalman_ti(u, y, L_stat, A, B, C);

a = (yhat_luenberg-y')';
b = (yhat_kalman_tv-y')';
c = (yhat_kalman_ti-y')';

figure(2); clf; sgtitle(filename); 
subplot(3,2,1); plot(1:1000, a, 'r'); yticks([-200 0 200]); ylabel('Luenberg') 
subplot(3,2,3); plot(1:1000, b, 'g'); yticks([-200 0 200]); ylabel('TV Kalman') 
subplot(3,2,5); plot(1:1000, c, 'b'); yticks([-200 0 200]); ylabel('TI Kalman') 
subplot(3,2,[2 4 6]); boxplot([a,b,c],'Colors',['r' 'g' 'b'],'Labels',{'L','TI K', 'TV K'}); 
yticks('auto')

MSE2 = [immse(yhat_luenberg(:), y(:)) ...
       immse(yhat_kalman_tv(:), y(:)) ...
       immse(yhat_kalman_ti(:), y(:))]

figure(4); hold on 
plot(1:1000, yhat_kalman_tv, 'r')
plot(1:1000, y, 'b')

%% DS3
filename = ['Dataset3.txt']
[u, y] = pablo_load_data(filename);
yhat_luenberg = pablo_luenberg(u, y, LuenbergObserverPoles, A, B, C);
[yhat_kalman_tv, L_stat, P_stat] = pablo_kalman_tv(u, y, A, B, C, LL, Q, R);
yhat_kalman_ti = pablo_kalman_ti(u, y, L_stat, A, B, C);

a = (yhat_luenberg-y')';
b = (yhat_kalman_tv-y')';
c = (yhat_kalman_ti-y')';

figure(3); clf; sgtitle(filename); 
line1 = subplot(3,2,1); plot(1:1000, a, 'r'); yticks([-50 0 50]); ylabel('Luenberg') 
line2 = subplot(3,2,3); plot(1:1000, b, 'g'); yticks([-50 0 50]); ylabel('TV Kalman') 
line3 = subplot(3,2,5); plot(1:1000, c, 'b'); yticks([-50 0 50]); ylabel('TI Kalman') 
line4 = subplot(3,2,[2 4 6]); boxplot([a,b,c],'Colors',['r' 'g' 'b'],'Labels',{'L','TI K', 'TV K'}); 
yticks([-120 -100 -80 -60 -40 -20 0 20 40 60 80 100 120])


MSE3 = [immse(yhat_luenberg, y') ...
       immse(yhat_kalman_tv, y') ...
       immse(yhat_kalman_ti, y')]

   

%% Summary of results
table(MSE1(:), MSE2(:), MSE3(:), 'VariableNames', {'Dataset 1','Dataset 2','Dataset 3'}, 'RowNames', {'Luenberg', 'Time Variant Kalman', 'Time Invariant Kalman'})
   
   
%% Functions from now on

function [u, y] = pablo_load_data(filename);
    %%% fancylooking one-liner for data loading 
    [u, y] = textread(filename,'%f %f','headerlines',1);
end


function yhat_luenberg = pablo_luenberg(u, y, P, A, B, C);
    %%% Gets a one-step prediction from Luembergs observer for the system 
    %%% output yhat. Function inputs are:
    %%%     u: known input
    %%%     y: known output
    %%%     P: observer poles
    %%%     A, B, C: propagation & measurement equation matrices*
 
    L = (place(A',C',P))';
    
    xhat = zeros(2,1,max(size(u)));
    yhat = zeros(1,max(size(u)));
    
    for k = 1:max(size(u))
        xhat(:,:,k+1) = A * xhat(:,:,k) + B*u(k) + L*(y(k) - C*xhat(:,:,k));
        yhat(k) = C * xhat(:,:,k);
    end
    
    xhat_luenberg = xhat;
    yhat_luenberg = yhat;
end


function [yhat_kalman_tv, converging_L, converging_P] ...
            = pablo_kalman_tv(u, y, A, B, C, LL, Q, R);
    %%% Gets a one-step prediction from Kalman's filter for the system 
    %%% output yhat. Also exports stationary & hopefully convergent values 
    %%% for L and P. Function inputs are:
    %%%     u: known input
    %%%     y: known output
    %%%     A, B, C: propagation & measurement equation matrices*
    %%%     LL: coefficient matrix for the process noise w
    %%%     Q, R: covariance matrices for process and masurement noise

    xhat = zeros(2,1,max(size(u)));
    yhat = zeros(1,max(size(u)));

    L = zeros(2,1,max(size(u)));  % init filter gain as zero
    P = zeros(2,2,max(size(u)));  % init covariance estimation error as zero
    
    % we are at instant K (known state) aiming to predict k+1
    for k = 1:max(size(u))-1
        % Propagation loop
        xhat(:,:,k+1) = A*xhat(:,:,k) + B*u(k);
        P(:,:,k+1) = A*P(k)*A' + LL*Q*LL';
        
        % Predict output. Needs to be extracted here to keep causality.
        yhat(k+1) = C*xhat(:,:,k+1);
        
        % Upgrade loop - Theoretically this would happen at the beggining 
        % of the next loop, therefore all (k+1)'s are actually (k)'s so 
        % causality holds. Only placed here for implementation ease
        L(:,:,k+1) = P(:,:,k+1) * C' * (C*P(:,:,k+1)*C' + R)^(-1);
        xhat(:,:,k+1) = xhat(:,:,k+1) + L(:,:,k+1)*(y(k+1) - C*xhat(:,:,k+1));
        P(:,:,k+1) = P(:,:,k+1) - L(:,:,k+1)*C*P(:,:,k+1);
    end    
    
    xhat_kalman_tv = xhat;
    yhat_kalman_tv = yhat;
    
    converging_L = L(:,:,900);
    converging_P = P(:,:,900);
end


function yhat_kalman_ti = pablo_kalman_ti(u, y, L, A, B, C);
    
    %%% Gets a one-step prediction from Kalman's filter for the system 
    %%% output yhat. L and P matrices are considered constant for the whole
    %%% time. Function inputs are:
    %%%     u: known input
    %%%     y: known output
    %%%     P: observer poles
    %%%     A, B, C: propagation & measurement equation matrices*
    %%%     LL: coefficient matrix for the process noise w
    %%%     Q, R: covariance matrices for process and masurement noise
   
    xhat = zeros(2,1,max(size(u)));
    yhat = zeros(1,max(size(u)));
    
    % we are at instant K (known state) aiming to predict k+1
    for k = 1:max(size(u))-1
        % Propagation loop
        xhat(:,:,k+1) = A*xhat(:,:,k) + B*u(k);
        
        % Predict output. Needs to be extracted here to keep causality.
        yhat(k+1) = C*xhat(:,:,k+1);
        
        % Upgrade loop
        xhat(:,:,k+1) = xhat(:,:,k+1) + L*(y(k+1) - C*xhat(:,:,k+1));
    end
    
    xhat_kalman_ti = xhat;
    yhat_kalman_ti = yhat;
end