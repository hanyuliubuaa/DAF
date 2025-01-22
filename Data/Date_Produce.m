clear

% Noises
Q = diag([1e-2^2, 1e-2^2, 1e-2^2]);
% Q = diag([2^2, 2^2, 2^2]);
% R = diag([10^2, 10^2, 10^2]);
R = diag([100^2, 10^2]);
% R = diag([4^2, 2^2]);

% Measurement interval
T = 0.03;

% Total length of the measurement sequence
N = 211;

% Length of the prior measurements
L_warm = 10;

% Number of measurement sequences of train and test datasets.
M_train = 10000;
M_test = 1000;

% Produce the train dataset.
X_train = zeros(M_train, N, 3);
Z_train = zeros(M_train, N, 2);

for times = 1:M_train
    times
    % True states
    x_true = zeros(3, N);
    x_true(:, 1) = mvnrnd([0, 0, 0], 100*eye(3))'; % Initial state
    for i = 2 : N
        x_true(:, i) = integ_Lorenz(x_true(:, i-1), [0 T]) + mvnrnd([0; 0; 0], Q)';
    end
    % Measurement
    z = zeros(2, N);
    for i = 1 : N
        z(:, i) = h(x_true(:, i)) + mvnrnd([0, 0], R)';
    end
    
    X_train(times, :, :) = x_true';
    Z_train(times, :, :) = z';
end

Train_data = Z_train(:, 1:(N-1), :);
Train_label = Z_train(:, (L_warm+2):N, :);
Train_gt = X_train(:, (L_warm+1):(N-1), :);
Train_true_filter = X_train(:, 1:(N-1), :);

% Test dataset
X_test = zeros(M_test, N, 3);
Z_test = zeros(M_test, N, 2);

for times = 1:M_test
    times
    % True states
    x_true = zeros(3, N);
    x_true(:, 1) = mvnrnd([0, 0, 0], 100*eye(3))'; % Initial state
    for i = 2 : N
        x_true(:, i) = integ_Lorenz(x_true(:, i-1), [0 T]) + mvnrnd([0; 0; 0], Q)';
    end
    % Measurement
    z = zeros(2, N);
    for i = 1 : N
        z(:, i) = h(x_true(:, i)) + mvnrnd([0, 0], R)';
    end
    
    X_test(times, :, :) = x_true';
    Z_test(times, :, :) = z';
end

Test_data = Z_test(:, 1:(N-1), :);
Test_label = Z_test(:, (L_warm+2):N, :);
Test_gt = X_test(:, (L_warm+1):(N-1), :);
Test_true_filter = X_test(:, 1:(N-1), :);

% Save
Train_data = reshape(Train_data, M_train, size(Train_data,2)*size(Train_data,3));
Train_label = reshape(Train_label, M_train, size(Train_label,2)*size(Train_label,3));
Train_gt = reshape(Train_gt, M_train, size(Train_gt,2)*size(Train_gt,3));
Train_true_filter = reshape(Train_true_filter, M_train, size(Train_true_filter,2)*size(Train_true_filter,3));
Test_data = reshape(Test_data, M_test, size(Test_data,2)*size(Test_data,3));
Test_label = reshape(Test_label, M_test, size(Test_label,2)*size(Test_label,3));
Test_gt = reshape(Test_gt, M_test, size(Test_gt,2)*size(Test_gt,3));
Test_true_filter = reshape(Test_true_filter, M_test, size(Test_true_filter,2)*size(Test_true_filter,3));
save('./case2/Train_data.txt', 'Train_data', '-ascii', '-double')
save('./case2/Train_label.txt', 'Train_label', '-ascii', '-double')
save('./case2/Train_gt.txt', 'Train_gt', '-ascii', '-double')
save('./case2/Train_true_filter.txt', 'Train_true_filter', '-ascii', '-double')
save('./case2/Test_data.txt', 'Test_data', '-ascii', '-double')
save('./case2/Test_label.txt', 'Test_label', '-ascii', '-double')
save('./case2/Test_gt.txt', 'Test_gt', '-ascii', '-double')
save('./case2/Test_true_filter.txt', 'Test_true_filter', '-ascii', '-double')