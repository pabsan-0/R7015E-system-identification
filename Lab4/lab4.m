format compact 
set(gca,'FontName','Times');

%% Data loading & misc
boom_angle = textread('./datasets/boom_angle_deg.txt','%f','headerlines',0);
boom_trate = textread('./datasets/boom_speed_deg_per_sec.txt','%f','headerlines',0);

bucket_angle = textread('./datasets/bucket_angle_deg.txt','%f','headerlines',0);
bucket_trate = textread('./datasets/bucket_speed_deg_per_sec.txt','%f','headerlines',0);

force_lift = textread('./datasets/force_lift_MN.txt','%f','headerlines',0);
force_tilt = textread('./datasets/force_tilt_MN.txt','%f','headerlines',0);

lift_joystick = textread('./datasets/lift_joystick.txt','%f','headerlines',0);
throttle = textread('./datasets/throttle.txt','%f','headerlines',0);
tilt_joystick = textread('./datasets/tilt_joystick.txt','%f','headerlines',0);

drive_revs = textread('./datasets/drive_axle_speed_rps.txt','%f','headerlines',0);
engine_revs = textread('./datasets/engine_speed_rps.txt','%f','headerlines',0);
gear_index = textread('./datasets/gear_index.txt','%f','headerlines',0);
digging = textread('./datasets/digging.txt','%f','headerlines',0);


all_data = [boom_angle(:) boom_trate(:) bucket_angle(:) bucket_trate(:)...
            force_lift(:) force_tilt(:) lift_joystick(:) throttle(:)...
            tilt_joystick(:) drive_revs(:) gear_index(:) digging(:) engine_revs(:)];


% Just defining a dictionary to hold the names in order
keySet = {1,2,3,4,5,6,7,8,9,10,11,12};
valueSet = {'boom_angle', 'boom_trate', 'bucket_angle', 'bucket_trate',...
            'force_lift', 'force_tilt', 'lift_joystick', 'throttle',...
            'tilt_joystick', 'drive_revs', 'gear_index', 'digging'};
names = containers.Map(keySet,valueSet);

% assembling the data in a big matrix
in_data = [boom_angle(:) boom_trate(:) bucket_angle(:) bucket_trate(:)...
            force_lift(:) force_tilt(:) lift_joystick(:) throttle(:)...
            tilt_joystick(:) drive_revs(:) gear_index(:) digging(:)];

% normalize the data as a whole for simplicity 
% (would be reasonable to normalize only training data & then applying 
% same normalziation  to the test one)
in_data = normalize(in_data, 1);
out_data = normalize(engine_revs);


%% Normalized splits 10000 + 4003
x_train = in_data(1:10000,:);
y_train = out_data(1:10000,:);

x_valid = in_data(10000:end,:);
y_valid = out_data(10000:end,:);


%% cov matrix of in_data (ofc we use training split)
Q = cov(x_train)


%% Studying the covariance of each variable w.r. to the output
% the closer to zero the more independent the variables are

% calculate covariances between each RV and the engine rps
Qy = [];
for i = 1:12
    crosscov = cov(x_train(:,i), y_train);
    crosscov = crosscov(1,2);
    Qy = [Qy; (crosscov)];
end

table(Qy,'RowNames',valueSet);

% Get sorting index for covariances
[~, index]= sort(abs(Qy),'descend');

% The following list holds the variables. The higher a variable is in the
% table the more linked it is to the engine_rps. To reduce system order
% start removing from the bottom ones.
names_by_relevance = containers.Map(keySet,valueSet(index)).values';
table(Qy(index), index,'VariableNames', {'Cov(each,engine_rps)','Idx'},'RowNames', names_by_relevance)



%% Visual templates dump
% array2table(Q, 'VariableNames', valueSet, 'RowNames', valueSet)
% array2table(Q, 'VariableNames', {'1','2','3','4','5','6','7','8','9','10','11','12'}, 'RowNames', {'1','2','3','4','5','6','7','8','9','10','11','12'})
% figure(); clf; hold on; scatter(x_train(:,2), x_train(:,7))


%%  QUICK TESTS OF ARX & ARMAX -> Upgrading to armax provides no extra benefit & math takes longer

%%% ARX
arxAA1_mse_train = [];
arxAA1_mse_valid = [];
for a = 1:10
    na = [a a a a a a a a a a a a];
    nb = [a a a a a a a a a a a a];
    nk = [1];
    arxAA1 = arx([y_train, x_train],[na nb nk]);
    arxAA1_mse_train = [arxAA1_mse_train  immse(y_train,sim(arxAA1, x_train))];
    arxAA1_mse_valid = [arxAA1_mse_valid  immse(y_valid,sim(arxAA1, x_valid))];
end

%%% ARMAX
armaxAAA1_mse_train = [];
armaxAAA1_mse_valid = [];
for a = 1:10
    na = [a];
    nb = [a a a a a a a a a a a a];
    nc = [a a a a a a a a a a a a];
    nk = [1];
    armaxAAA1 = armax([y_train, x_train],[na nb nc nk]);

    armaxAAA1_mse_train = [armaxAAA1_mse_train  immse(y_train, sim(armaxAAA1, x_train))];
    armaxAAA1_mse_valid = [armaxAAA1_mse_valid  immse(y_valid, sim(armaxAAA1, x_valid))];
end

figure(); clf; 
subplot(1,2,1); hold on; stem(arxAA1_mse_valid); stem(arxAA1_mse_train); title('ARX')
legend('Validation MSE','Training MSE'); xlabel('Model order')
subplot(1,2,2); hold on; stem(armaxAAA1_mse_valid); stem(armaxAAA1_mse_train); title('ARMAX')
legend('Validation MSE','Training MSE'); xlabel('Model order')


%% Ablation experiments - removing features
best_mse_arx_vanilla = min(arxAA1_mse_valid);


%%% Removing the top covariance element
a = 2; nk = [1];
x_train_ = x_train(:, [1 1 1 1   1 1 1 0   1 1 1 1] == 1);
x_valid_ = x_valid(:, [1 1 1 1   1 1 1 0   1 1 1 1] == 1);
na = [1 1 1 1   1 1 1  1 1 1 1] * a;
nb = [1 1 1 1   1 1 1  1 1 1 1] * a;
headless_mse_valid = immse(y_valid, sim(arx([y_train, x_train_],[na nb nk]), x_valid_));


%%% Removing the least covariance element
a = 2; nk = [1];
x_train_ = x_train(:, [1 1 1 1   1 0 1 1   1 1 1 1] == 1);
x_valid_ = x_valid(:, [1 1 1 1   1 0 1 1   1 1 1 1] == 1);
na = [1 1 1 1   1 1 1 1   1 1 1] * a;
nb = [1 1 1 1   1 1 1 1   1 1 1] * a;
bottomless_mse_valid = immse(y_valid, sim(arx([y_train, x_train_],[na nb nk]), x_valid_));


%%% Keeping the 3 best cov elements
a = 2; nk = [1];
x_train_ = x_train(:, [0 0 0 0   1 0 0 1   0 0 0 1] == 1);
x_valid_ = x_valid(:, [0 0 0 0   1 0 0 1   0 0 0 1] == 1);
na = [1 1 1] * a;
nb = [1 1 1] * a;
best3_mse_valid = immse(y_valid, sim(arx([y_train, x_train_],[na nb nk]), x_valid_));


%%% Removing the 3 worst covariance elements
a = 2; nk = [1];
x_train_ = x_train(:, [1 1 1 1   0 0 1 1   1 1 0 1] == 1);
x_valid_ = x_valid(:, [1 1 1 1   0 0 1 1   1 1 0 1] == 1);
na = ([1 1 1 1  1 1   1 1 1] == 1) * a;
nb = ([1 1 1 1  1 1   1 1 1] == 1) * a;
worst3_mse_valid = immse(y_valid, sim(arx([y_train, x_train_],[na nb nk]), x_valid_));


% figure(); hold on
% stem([headless_mse_valid bottomless_mse_valid best3_mse_valid worst3_mse_valid])
% line([-1,best_mse_arx_vanilla], [2, best_mse_arx_vanilla])

all_mses = [best_mse_arx_vanilla headless_mse_valid bottomless_mse_valid best3_mse_valid worst3_mse_valid];
table(all_mses(:), 'VariableNames', {'ARX221 MSE'},'RowNames', {'Baseline','Remove highest cov(feat, y)','Remove lowest cov(feat, y) ','Keep only 3 highest cov(feat, y)','Remove 3 lowest cov(feat, y)'})



%% After analysis of the above
% Removing elements in the order given by the cov(x_i, y) table
% Reminder: 'index' holds the ordered indices

a = 2;
figure(); clf; sgtitle('True vs predicted output for each number of considered features')
mses = []; means = []; variances = [];
for i = 1:length(index)
    vec = choice_vector(index, i);
    x_train_ = x_train(:, vec);
    x_valid_ = x_valid(:, vec);
    
    nk = [1];
    na = ones(1, sum(vec)) * a;
    nb = ones(1, sum(vec)) * a;
    
    model = arx([y_train, x_train_],[na nb nk]);
    y_pred = sim(model, x_valid_);
    
    subplot(6,2,i); hold on; plot(y_pred); plot(y_valid); xlim([1000 1500]); title(strcat(string(i), ' features'))
    
    mses =      [mses       immse(y_valid, y_pred)];
    means =     [means      mean(y_valid - y_pred)];
    variances = [variances  var(y_valid - y_pred)];
end

figure(); clf; hold on
stem(mses)
ylabel('Validation MSE for ARX221 model')
xlabel('Number of input channels (sorted by cov(xi, y))')

table(Qy(index), index, mses(:), means(:), variances(:),'VariableNames', {'Cov(feat,y)','Index', 'ARX221 MSE', 'ARX221 Mean error', 'ARX221 Variance of error'},'RowNames', names_by_relevance)

%% final model decision ARX221 w/ 3 input
num_features = 3;
model_order = 2;
vec = choice_vector(index, num_features);
x_train_ = x_train(:, vec);
x_valid_ = x_valid(:, vec);

nk = [1];
na = ones(1, sum(vec)) * model_order;
nb = ones(1, sum(vec)) * model_order;

model = arx([y_train, x_train_],[na nb nk]);
y_pred = sim(model, x_valid_);

figure(); clf; hold on
plot(y_pred); plot(y_valid)
legend('Predicted output','Real output')
xlabel('Time')
ylabel('Engine RPS')

%% testing choice_vector function
choice_vector(index, 1);

function vec = choice_vector(index, number_to_keep)
    %%% This function takes the vector that sorts each input according to
    %%% their covariance with the output & produces a one-hot vector (BoW)
    %%% encoding the position of the NUMBER_TO_KEEP more significant ones
    %%% in the big matrix with all the elements.
    
    onehot = eye(length(index));
    vec = zeros(size(index));
    for i = 1:number_to_keep
        vec = vec + onehot(:, index(i));
    end
    vec = vec == 1;
    vec = vec(:)';
end
