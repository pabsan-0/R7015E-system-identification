format compact

%% Loading and preview
co2 = textread('./datasets_training/training-CO2.txt','%f','headerlines',0);
vnt = textread('./datasets_training/training-ventilation.txt','%f','headerlines',0);
occ = textread('./datasets_training/training-occupancy.txt','%f','headerlines',0);

figure(1); clf; hold on; set(gca,'FontName','Times')
plot(co2); plot(vnt); plot(occ);
legend('CO2','ventilation','Occupancy')


%% Laying out the ML problem

% Input & ouput of the system' arx model
u = [co2(1:end-1)'; vnt(1:end-1)'; occ(1:end-1)'];
y =  co2(2:end)';

% Theta = [a, bu, bo, s2];
theta_0_guess = [1, 0, 1, 1];

% Apply constrained minimization of the negative loglikelihood of the error
flh = @(x) loglikelihood(x,y,u);            % function handle for syntax

[theta, fval] = fmincon(flh, theta_0_guess,...
                        [],[],[],[],...
                        [0,-inf,0,0],...    % lower bounds
                        [1,0,inf,inf]);     % upper bounds

a = theta(1);
bu = theta(2);
bo = theta(3);
s2 = theta(4);
  

%% Validating the result...
           
% % Do i overfit on training dataset? i should
% % Predict and post-treat prediction
% occ_pred = (co2(2:end) - a*co2(1:end-1) - bu*vnt(1:end-1))/bo;
% 
% % Round + Relu + smoothing
% occ_pred = round_relu_bartlett_smoothing(occ_pred, 10);
% 
% % Allow changes only on door opening
% % occ_pred = allow_changes_on_diff(occ_pred, occ);
% 
% figure(2); clf; hold on
% plot(occ)
% stem(occ_pred)
% legend('measured occ','predicted occ')
% sgtitle('This for experimental data')

% Validate on test data
co2 = textread('./datasets_testing/testing-CO2.txt','%f','headerlines',0);
vnt = textread('./datasets_testing/testing-ventilation.txt','%f','headerlines',0);
occ = textread('./datasets_testing/testing-occupancy.txt','%f','headerlines',0);

% Compute the prediction based on the model and data
occ_pred = (co2(2:end) - a*co2(1:end-1) - bu*vnt(1:end-1))/bo;

% Bartlett smoothing
occ_pred = round_relu_bartlett_smoothing(occ_pred, 10)


figure(3); clf; set(gca,'FontName','Times')
sgtitle('Model performance on testing data', 'FontName', 'Times')
%
subplot(2,1,1); hold on;  set(gca,'FontName','Times')
h = area(occ); h.FaceColor = [.7 .7 .7];
plot(occ_pred,  'LineWidth', 1, 'color', [0 0 1])
legend('Measured occupancy','Predicted occupancy')
ylabel('Occupancy'); xlabel('Time')
%
subplot(2,1,2); hold on;  set(gca,'FontName','Times')
area((occ - occ_pred(1:length(occ))))
legend('Prediction error')
ylabel('Occupancy error'); xlabel('Time')


%%
function [loglikelihood] = loglikelihood(x,y,u)
    %%% This is a modification of Robert Hedman's built function to
    %%% compute a prediction error then its loglikelihood from an
    %%% arbitrary vector input theta.

    % theta = [a, bu, bo, e_variance];
    N = numel(y);
    yhat = x(1)*u(1,:) + x(2)*u(2,:) + x(3)*u(3,:);
    e = y - yhat;
    
    loglikelihood = N/2*log(2*pi) + N/2*log(x(4)) + (2*x(4))^-1 * sum((e-mean(e)).^2); 
end

function signal = round_relu_bartlett_smoothing(signal, width)
    signal = round(max(conv(bartlett(width),signal)/sum(bartlett(width)),0))
end

function signal_out = allow_changes_on_diff(signal_in, pattern)
    pattern_deriv = diff(pattern);
    signal_deriv = diff(signal_in);
    
    changes_allowed = 0
    signal_out = 0 .* signal_in;
    signal_out = signal_in(1);
    
    for i = 2:length(pattern_deriv)-1
        if pattern_deriv(i) ~= 0
            changes_allowed = 1;
        end
        if changes_allowed == 1 & signal_deriv(i)~=0 
            signal_out(i) = signal_in(i);
            changes_alowed = 0;
        else
            signal_out(i) = signal_out(i-1);
        end
    end
end