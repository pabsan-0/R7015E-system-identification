format compact
clear
%% load stuff
Assignment2_Feb_slave; % Load ABCD statespace from real sys
load('Dataset4.mat')             % Load data

C = [1,0,0,0; 0,0,1,0]; % override C for full observability
system = c2d(ss(A,B,C,D),fSamplingPeriod);

Ad = system.A;
Bd = system.B;
Cd = system.C;
Dd = system.D;

my_poles = pole(system);

% Initial state-vector is always x0 = [0; 0];
x0 = [0;0;0;0];

% Define I/O
u = aafProcessedInformation(U_INDEX,:);
y = [aafProcessedInformation(MEASURED_X_W_INDEX,:);...
     aafProcessedInformation(MEASURED_THETA_B_INDEX,:)];

 

%% Time variant kalman
fprintf('### Computing TV Kalman... \n')
Q = eye(4);    % cov process noise
R = zeros(2);    % cov measurement noise
R(1,1) = var(y(1,:));
R(2,2) = var(y(2,:))

[yhat, mse] = custom_tvkalman(u, y, Ad, Bd, Cd, Dd, Q, R, [], []);
mse_KV = mse;
mse_KV_1 = [mse_KV; sqrt(mse_KV(1)^2 + mse_KV(2)^2)];
e1 = yhat-y;

figure(2); clf; hold on
plot(y(1,:)); plot(yhat(1,:));
plot(y(2,:)); plot(yhat(2,:));
legend('y1','y1hat','y2','y2hat')


%% Montecarlo algorithm to try and optimize the kalman filter TIME VARIANT

u_ = u(1:100)
y_ = y(:,1:100)
mse_hold = 99;

Q_src = Q;
R_src = R;

for i = 1:20000
    Q = Q_src .* normrnd([1 1 1 1], 0.1);
    R = R_src .* normrnd([1 1], 0.1);
    
    [~, mse] = custom_tvkalman(u_, y_, Ad, Bd, Cd, Dd, Q, R, [], []);
    mse = sqrt(mse(1)^2 + mse(2)^2);
    
    if mse < mse_hold
        Q_src = Q;
        R_src = R;
        mse_hold = mse;
        fprintf('Improved!')
    end
    
    fprintf(string(i)); fprintf('\n')
end

[yhat, mse] = custom_tvkalman(u, y, Ad, Bd, Cd, Dd, Q, R, [], []);
mse_KV = mse;
mse_KV_2 = [mse_KV; sqrt(mse_KV(1)^2 + mse_KV(2)^2)];
e2 = yhat-y;

Q_opt_tv = Q
R_opt_tv = R

figure(1); clf; sgtitle('Time variant Kalman errors'); 
subplot(1,2,1); boxplot([e1(1,:)',e1(2,:)'],'Colors',['r' 'b'],'Labels',{'X_W','THETA_B'}); title('Baseline')
subplot(1,2,2); boxplot([e2(1,:)',e2(2,:)'],'Colors',['r' 'b'],'Labels',{'X_W','THETA_B'}); title('After tuning')
yticks()



% %%% Time invariant kalman (generic, just here as template)
% fprintf('### Computing TI Kalman... \n')
% Q = eye(4)      % cov process noise
% R = zeros(2);    % cov measurement noise
% R(1,1) = var(y(1,:));
% R(2,2) = var(y(2,:)) 
% 
% [yhat, mse] = custom_tikalman(u, y, Ad, Bd, Cd, Dd, Q, R);
% mse_KI = mse;
% mse_KI_1 = [mse_KI; sqrt(mse_KI(1)^2 + mse_KI(2)^2)];
% e1 = yhat-y;
% 
% figure(2); clf; hold on
% plot(y(1,:)); plot(yhat(1,:));
% plot(y(2,:)); plot(yhat(2,:));
% legend('y1','y1hat','y2','y2hat')


%% Montecarlo algorithm to try and optimize the kalman filter TIME INVARIANT

u_ = u(1:100);
y_ = y(:,1:100);
mse_hold = 99;

Q_src = Q
R_src = R

for i = 1:20000
    Q = Q_src .* normrnd([1 1 1 1], 0.1);
    R = R_src .* normrnd([1 1], 0.1);
    
    [~, mse] = custom_tikalman(u_, y_, Ad, Bd, Cd, Dd, Q, R);
    mse = sqrt(mse(1).^2 + mse(2).^2);
    
    if mse < mse_hold
        Q_src = Q;
        R_src = R;
        mse_hold = mse;
        fprintf('Improved!')
    end
    fprintf(string(i)); fprintf('\n')
end

[yhat, mse] = custom_tikalman(u, y, Ad, Bd, Cd, Dd, Q, R);
mse_KI = mse;
mse_KI_2 = [mse_KI; sqrt(mse_KI(1)^2 + mse_KI(2)^2)];
e2 = yhat-y;

Q_opt_ti = Q
R_opt_ti = R

figure(2); clf; sgtitle('Time invariant Kalman errors'); 
subplot(1,2,1); boxplot([e1(1,:)',e1(2,:)'],'Colors',['r' 'b'],'Labels',{'X_W','THETA_B'}); title('Baseline')
subplot(1,2,2); boxplot([e2(1,:)',e2(2,:)'],'Colors',['r' 'b'],'Labels',{'X_W','THETA_B'}); title('After tuning')
yticks()

%%
table([mse_KV_1 mse_KV_2], [mse_KI_1 mse_KI_2], 'VariableNames',{'TV Kalman','TI Kalman'}, 'RowNames', {'MSE X_W','MSE THETA_B','Combined MSE'})
table(mse_KV_1, mse_KV_2, mse_KI_1, mse_KI_2, 'VariableNames',{'Baseline','After tuning','Baselin','Afte tuning'}, 'RowNames', {'MSE X_W','MSE THETA_B','Combined MSE'})

fprintf('\n### Optimal matrices Q and R for each case are...\n')
Q_opt_tv
R_opt_tv
Q_opt_ti
R_opt_ti


%%
function [yhat, mse] = custom_tvkalman(u, y, A, B, C, D, Q, R, L0, P0)
    %%% Time variant Kalman filter
    % Q: Covariance matrix for process noise
    % R: Covariance matrix for measurement noise
    % L0: initial filter gain value - initiated within the function
    % P0: initial covariance estimation error - initiated within the function
 
    xhat = zeros(4,max(size(u)));
    yhat = zeros(2,max(size(u)));

    L = zeros(4,2,max(size(u)));  % filter gain K
    P = zeros(4,4,max(size(u)));  % covariance estimation error
    LL = eye(4);                  % process noise coefficient
    
    % we are at instant K (known state) aiming to predict k+1
    for k = 1:max(size(u))-1
        % Propagation loop
        xhat(:,k+1) = A*xhat(:,k) + B*u(k);
        P(:,:,k+1)  = A*P(:,:,k)*A' + LL*Q*LL';

        % Predict output. Needs to be extracted here to keep causality.
        yhat(:,k+1) = C*xhat(:,k+1);

        % Upgrade loop - This would happen at the beggining of the next
        % loop, therefore all (k+1)s are actually (k)s so causality holds.
        L(:,:,k+1)    = P(:,:,k+1) * C' * (C*P(:,:,k+1)*C' + R)^(-1);
        xhat(:,k+1) = xhat(:,k+1) + L(:,:,k+1)*(y(:,k+1) - C*xhat(:,k+1));
        P(:,:,k+1)  = P(:,:,k+1) - L(:,:,k+1)*C*P(:,:,k+1);
    end 
    
    mse = [immse(y(1,:), yhat(1,:)); immse(y(2,:), yhat(2,:))];
end


function [yhat, mse] = custom_tikalman(u, y, Ad, Bd, Cd, Dd, Q, R)
    %%% Time Invariant Kalman filter
    % Q: Covariance matrix for process noise
    % R: Covariance matrix for measurement noise

    xhat = zeros(4,max(size(u)));
    yhat = zeros(2,max(size(u)));
    
    S = [0 0;0 0;0 0;0 0];   % Covariance w,v == 0,0
    N = eye(length(S));     % It is multiplied by S = (0,0) so we do not care.
    [P, ~, ~] = idare(Ad', Cd', Q, R);
    L =(Ad*P*Cd' + N*S)/(Cd*P*Cd' + R);

    % we are at instant K (known state) aiming to predict k+1
    for k = 1:max(size(u))-1
        % Propagation loop
        xhat(:,k+1) = Ad*xhat(:,k) + Bd*u(k);

        % Predict output. Needs to be extracted here to keep causality.
        yhat(:,k+1) = Cd*xhat(:,k+1);

        % Upgrade loop
        xhat(:,k+1) = xhat(:,k+1) + L*(y(:,k+1) - Cd*xhat(:,k+1));
    end
    
    mse = [immse(y(1,:), yhat(1,:)); immse(y(2,:), yhat(2,:))];
end
