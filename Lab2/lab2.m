set(gca,'FontName','Times')
[u, y] = textread('Dataset1.txt','%f %f','headerlines',1);
ts = 0.08 % s

% Plot input and output for overview
figure(1); clf; hold on
set(gca,'FontName','Times')
plot(u); plot(y)
legend('Input', 'Output')
sgtitle('Input and output')

%% Correlation analysis -> model == h

% Running model extraction for different system orders
mses = []
for sys_order = 1:15
    [y_pred, h] = pablo_correlation(u, y, sys_order);
    mses = [mses; (1/length(y))*sum((y-y_pred).^2)];
end

% Running for a given system order
order = 999
[ypred, h] = pablo_correlation(u, y, order);
mse = (1/length(y))*sum((ypred - y).^2);

% Plot everything
figure(2); clf; hold on; set(gca,'FontName','Times')
subplot(1,2,1)
stem(mses); xlabel('System order' ,'FontName', 'Times'); ylabel('MSE(ypred, y)' ,'FontName', 'Times')
title('MSEs for different system orders' ,'FontName', 'Times')
%
subplot(1,2,2); hold on
plot(y,'b'); plot(ypred,'r');
legend('Real Output', 'Modelled systems output' ,'FontName', 'Times')
title(["Time-series for model order " + order] ,'FontName', 'Times')
sgtitle('Correlation analysis' ,'FontName', 'Times')




%% Raw ETFE  -> model == g_raw

% Frequency domain transfer func
graw = fft(u).\fft(y);  

% Back to time domain
gg = ifft(graw);           
gg = gg(1:16)               % Denoising (remove zeroish terms)
y_pred = filter(gg,1,u);    % Convolve input for a test

% Plotting
figure(3); clf; hold on
set(gca,'FontName','Times')
plot(y,'b'); plot(ypred,'r');
legend('Real Output', 'Modelled systems output')
sgtitle('ETFE model prediction vs reality')



%% Smoothed ETFE 

% Defining the problem of predicting y...
mse = []
varc = []
for width = 1:1:40
    % Obtaining time-domain TF
    gsmooth = conv(bartlett(width),graw.*abs(fft(u)).^2)...
       ./ conv(bartlett(width),abs(fft(u)).^2);
    gsmooth = gsmooth(2:end-1) % remove nans at 1st and end element
    gg = abs(ifft(gsmooth));
    
    % Prediction of output
    y_pred = filter(gg,1,u);  
    % bias = [bias mean(y_pred) - mean(y)];
    mse  = [mse mean((y_pred(:) - y(:)).^2)];
    varc = [varc var(y_pred, y)]
end

% Bias, variance and mse 
figure(5); clf; hold on
set(gca,'FontName','Times')
plot(varc); plot(mse); plot(sqrt(abs(mse(:)-varc(:))))
legend('Variance','MSE','Bias')
xlabel('Barlett window width')
sgtitle('MSE, bias and variance for different smoothing gains')


% min mse
w_opt = find(mse == min(mse)) 

% Freq. domain calculation of smoothed TF
width = w_opt
gsmooth = conv(bartlett(width),graw.*abs(fft(u)).^2)...
       ./ conv(bartlett(width),abs(fft(u)).^2);
gsmooth = gsmooth(~isnan(gsmooth));  % remove nans at 1st and end element

% Back to time domain & compare gsmooth to gnoisy from previous section
gg = ifft(gsmooth);

figure(6); clf; hold on
set(gca,'FontName','Times')
plot(abs(gg)); 
plot(ifft(fft(u).\fft(y))) 
legend('Softened TF', 'Raw TF')
sgtitle('Predictor values for MMSE with smoothing')




%% Bode plotting

figure();set(gca,'FontName','Times')
hold on;
subplot(1,3,1); semilogx(20*log10(abs(fft(h)))); title('From correlation analysis' ,'FontName', 'Times')
ylabel('Gain' ,'FontName', 'Times'); xlabel('Frequency' ,'FontName', 'Times')
subplot(1,3,2); semilogx(20*log10(abs(graw))); title('From ETFE'); xlabel('Frequency' ,'FontName', 'Times')
subplot(1,3,3); semilogx(20*log10(abs(gsmooth))); title('From smoothed ETFE' ,'FontName', 'Times'); xlabel('Frequency' ,'FontName', 'Times')
sgtitle('Bode plots for the estimated transfer functions' ,'FontName', 'Times')


figure();set(gca,'FontName','Times')
hold on;
subplot(1,3,1); plot(h); title('From correlation analysis' ,'FontName', 'Times')
ylabel('Value' ,'FontName', 'Times'); xlabel('nth coefficient' ,'FontName', 'Times')
subplot(1,3,2); plot(abs(ifft(graw))); title('From ETFE' ,'FontName', 'Times'); xlabel('nth coefficient' ,'FontName', 'Times')
subplot(1,3,3); plot(abs(ifft(gsmooth))); title('From smoothed ETFE' ,'FontName', 'Times'); xlabel('nth coefficient' ,'FontName', 'Times')
sgtitle('Vectorized model for the estimated transfer functions' ,'FontName', 'Times')

%%
[y_pred,~] = pablo_correlation(u, y, 999); mse_corr_full = immse(y_pred(:), y(:))
[y_pred,~] = pablo_correlation(u, y, 10); mse_corr_10 = immse(y_pred(:), y(:))
[y_pred,~] = pablo_correlation(u, y, 50); mse_corr_50 = immse(y_pred(:), y(:))


hraw = abs(ifft(graw));
y_pred = filter(hraw,1,u);       mse_raw_full = immse(y_pred(:), y(:))
y_pred = filter(hraw(1:10),1,u); mse_raw_10 = immse(y_pred(:), y(:))
y_pred = filter(hraw(1:50),1,u); mse_raw_50 = immse(y_pred(:), y(:))

hsmooth = abs(ifft(gsmooth));
y_pred = filter(hsmooth,1,u);       mse_smooth_full = immse(y_pred(:), y(:))
y_pred = filter(hsmooth(1:10),1,u); mse_smooth_10 = immse(y_pred(:), y(:))
y_pred = filter(hsmooth(1:50),1,u); mse_smooth_50 = immse(y_pred(:), y(:))


%%


function [y_pred, h] = pablo_correlation(x, y, sys_order)
    %%% Builds a zero-step-ahead predictor for a Moving Average process by
    %%% correlation analysis. In other words, correlations are  calculated 
    %%% from given input and output  and a h vector is obtained so that the
    %%% convolution {x * h = y} holds.
    
    % Building the Rxx matrix
    [rxx, ~] = xcorr(x);
    rxx = toeplitz(rxx(1000:1000+sys_order))

    % Building the Rxy vector
    [ryx, ~] = xcorr(y,x);
    ryx = ryx(1000:1000+sys_order)
    ryx = ryx(:);                   % must be column vector

    % Solving for H vector; because ry=rx*H...
    h = rxx \ ryx;
    
    y_pred = filter(h,1,x);
end