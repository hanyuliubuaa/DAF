clear
RMSE_UKF_1 = load('RMSE_UKF_1.txt')';
RMSE_EKF_1 = load('RMSE_EKF_1.txt')';
RMSE_PF_1_1 = load('RMSE_PF_1_1.txt')';
RMSE_PF_2_1 = load('RMSE_PF_2_1.txt')';
RMSE_CRLB_1 = load('RMSE_CRLB_1.txt')';
RMSE_DANSE_1 = load('RMSE_DANSE_1.txt')';
RMSE_PDAF_1 = load('RMSE_PDAF_1.txt')';

figure;
t = 1:200;
plot(t,RMSE_CRLB_1,'c', 'Marker', '*', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
hold on;
plot(t,RMSE_EKF_1,'k', 'Marker', 's', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_UKF_1,'r', 'Marker', 'o', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_PF_1_1,'m', 'Marker', '^', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_PF_2_1,'g', 'Marker', 'v', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_DANSE_1, 'Color', [1, 0.7, 0], 'Marker', 'x', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_PDAF_1,'b', 'Marker', 'p', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
legend('PCRLB', 'EKF', 'UKF', 'PF-100', 'PF-500', 'DANSE', 'DAF (ours)', 'FontName', 'Times New Roman')
xlim([1,200])
ylim([1, 8])
xlabel('Time (\itk\rm)', 'FontName', 'Times New Roman')
ylabel('RMSE', 'FontName', 'Times New Roman')

norm_RMSE_UKF_1 = sqrt(1/200*norm(RMSE_UKF_1(1:200))^2);
norm_RMSE_EKF_1 = sqrt(1/200*norm(RMSE_EKF_1(1:200))^2);
norm_RMSE_CRLB_1 = sqrt(1/200*norm(RMSE_CRLB_1(1:200))^2);
norm_RMSE_PDAF_1 = sqrt(1/200*norm(RMSE_PDAF_1(1:200))^2);
norm_RMSE_PF_1_1 = sqrt(1/200*norm(RMSE_PF_1_1(1:200))^2);
norm_RMSE_PF_2_1 = sqrt(1/200*norm(RMSE_PF_2_1(1:200))^2);
norm_RMSE_DANSE_1 = sqrt(1/200*norm(RMSE_DANSE_1(1:200))^2);


RMSE_UKF_2 = load('RMSE_UKF_2.txt')';
RMSE_EKF_2 = load('RMSE_EKF_2.txt')';
RMSE_PF_1_2 = load('RMSE_PF_1_2.txt')';
RMSE_PF_2_2 = load('RMSE_PF_2_2.txt')';
RMSE_CRLB_2 = load('RMSE_CRLB_2.txt')';
RMSE_DANSE_2 = load('RMSE_DANSE_2.txt')';
RMSE_PDAF_2 = load('RMSE_PDAF_2.txt')';

figure;
t = 1:200;
plot(t,RMSE_CRLB_2,'c', 'Marker', '*', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
hold on;
plot(t,RMSE_EKF_2,'k', 'Marker', 's', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_UKF_2,'r', 'Marker', 'o', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_PF_1_2,'m', 'Marker', '^', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_PF_2_2,'g', 'Marker', 'v', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
plot(t,RMSE_PDAF_2,'b', 'Marker', 'p', 'MarkerIndices', 20:20:length(t), 'MarkerSize', 8, 'linewidth', 1);
legend('PCRLB', 'EKF', 'UKF', 'PF-100', 'PF-500', 'DAF (ours)', 'FontName', 'Times New Roman')
xlim([1,200])
ylim([1, 9])
xlabel('Time (\itk\rm)', 'FontName', 'Times New Roman')
ylabel('RMSE', 'FontName', 'Times New Roman')

norm_RMSE_UKF_2 = sqrt(1/100*norm(RMSE_UKF_2(101:200))^2);
norm_RMSE_EKF_2 = sqrt(1/100*norm(RMSE_EKF_2(101:200))^2);
norm_RMSE_CRLB_2 = sqrt(1/100*norm(RMSE_CRLB_2(101:200))^2);
norm_RMSE_PDAF_2 = sqrt(1/100*norm(RMSE_PDAF_2(101:200))^2);
norm_RMSE_PF_1_2 = sqrt(1/100*norm(RMSE_PF_1_2(101:200))^2);
norm_RMSE_PF_2_2 = sqrt(1/100*norm(RMSE_PF_2_2(101:200))^2);
norm_RMSE_DANSE_2 = sqrt(1/100*norm(RMSE_DANSE_2(101:200))^2);