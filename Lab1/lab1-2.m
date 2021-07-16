format compact
clear
%% load stuff
lab1-2-slave.m % Load ABCD statespace from real sys
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

 
%% Luenberg
fprintf('\n### Computing Luenberg... \n')
L_poles = [0.3 0.4 0.3j -0.3j]'
% L_poles = [0.9 0.85 0.9j -0.9j]'

[yhat, mse] = custom_luenberg(u, y, Ad, Bd, Cd, Dd, L_poles);
mse_L = mse;
e1 = yhat-y

figure(1); clf; hold on
plot(y(1,:)); plot(yhat(1,:));
plot(y(2,:)); plot(yhat(2,:));
legend('y1','y1hat','y2','y2hat')



%% Time variant kalman
fprintf('### Computing TV Kalman... \n')
Q = eye(4)      % cov process noise
R = zeros(2);    % cov measurement noise
R(1,1) = var(y(1,:));
R(2,2) = var(y(2,:))

[yhat, mse] = custom_tvkalman(u, y, Ad, Bd, Cd, Dd, Q, R, [], []);
mse_KV = mse;
e2 = yhat-y

figure(2); clf; hold on
plot(y(1,:)); plot(yhat(1,:));
plot(y(2,:)); plot(yhat(2,:));
legend('y1','y1hat','y2','y2hat')



%% Time invariant kalman
fprintf('### Computing TI Kalman... \n')
Q = eye(4)      % cov process noise
R = zeros(2);    % cov measurement noise
R(1,1) = var(y(1,:));
R(2,2) = var(y(2,:)) 

[yhat, mse] = custom_tikalman(u, y, Ad, Bd, Cd, Dd, Q, R);
mse_KI = mse;
e3 = yhat-y

figure(3); clf; hold on
plot(y(1,:)); plot(yhat(1,:));
plot(y(2,:)); plot(yhat(2,:));
legend('y1','y1hat','y2','y2hat')

mses = [mse_L mse_KV mse_KI];

%%
mse_L  = [mse_L;  sqrt(mse_L(1)^2  + mse_L(2)^2)]
mse_KV = [mse_KV; sqrt(mse_KV(1)^2 + mse_KV(2)^2)]
mse_KI = [mse_KI; sqrt(mse_KI(1)^2 + mse_KI(2)^2)]


figure(1); clf; sgtitle('Dataset 4: Error for X_W'); 
subplot(3,2,1); plot(1:12000, e1(1,:), 'r'); yticks(); ylabel('Luenberg') 
subplot(3,2,3); plot(1:12000, e2(1,:), 'g'); yticks(); ylabel('TV Kalman') 
subplot(3,2,5); plot(1:12000, e3(1,:), 'b'); yticks(); ylabel('TI Kalman') 
subplot(3,2,[2 4 6]); boxplot([e1(1,:)',e2(1,:)',e3(1,:)'],'Colors',['r' 'g' 'b'],'Labels',{'L','TI K', 'TV K'}); 
yticks()

figure(2); clf; sgtitle('Dataset 4: Error for THETA_B'); 
subplot(3,2,1); plot(1:12000, e1(2,:), 'r'); yticks(); ylabel('Luenberg') 
subplot(3,2,3); plot(1:12000, e2(2,:), 'g'); yticks(); ylabel('TV Kalman') 
subplot(3,2,5); plot(1:12000, e3(2,:), 'b'); yticks(); ylabel('TI Kalman') 
subplot(3,2,[2 4 6]); boxplot([e1(2,:)',e2(2,:)',e3(2,:)'],'Colors',['r' 'g' 'b'],'Labels',{'L','TI K', 'TV K'}); 
yticks()

figure(3); clf; sgtitle('Dataset 4: Combined error'); 
subplot(3,1,1); plot(1:12000, (e1(1,:).^2 + e1(2,:).^2).^(1/2), 'r'); yticks(); ylabel('Luenberg') 
subplot(3,1,2); plot(1:12000, (e2(1,:).^2 + e2(2,:).^2).^(1/2), 'g'); yticks(); ylabel('TV Kalman') 
subplot(3,1,3); plot(1:12000, (e3(1,:).^2 + e3(2,:).^2).^(1/2), 'b'); yticks(); ylabel('TI Kalman') 


table(mse_L, mse_KV, mse_KI, 'VariableNames',{'Luenberg','Kalman TV','Kalman TI'}, 'RowNames', {'MSE Y1','MSE Y2','||MSE Y1, MSE Y2||_2'})


%%
function  [yhat, mse] = custom_luenberg(u, y, Ad, Bd, Cd, Dd, L_poles)
    %%% Luenberg observer
    L = place(Ad', Cd', L_poles)';

    xhat = zeros(4,max(size(u)));
    yhat = zeros(2,max(size(u))); 

    for k = 1:max(size(u))
        xhat(:,k+1) = Ad*xhat(:,k) + Bd*u(k) + L*(y(:,k)-Cd*xhat(:,k));
        yhat(:,k+1)   = Cd*xhat(:,k+1);
    end
    
    yhat = yhat(:,1:end-1);
    mse = [immse(y(1,:), yhat(1,:)); immse(y(2,:), yhat(2,:))];
end


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
    
    S = [0 0;0 0;0 0;0 0]   % Covariance w,v == 0,0
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
