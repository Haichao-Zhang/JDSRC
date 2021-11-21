%%=========================================================================
%               demo code illustrating the usage of 
%       the Joint Dynamic Sparse Representation algorithm
%                    by Haichao Zhang
%                   hczhang1@gmail.com
%                      Feb. 2012
%%=========================================================================



%% Synthetic data generation
N = 100; % number of training samples
d = 10;  % feature dim. for one task
K = 2;   % number of tasks

% dictionary 
X1 = randn(d, N);
X2 = randn(d, N);
X = [X1; X2];
task_ind = [0, d, 2*d];
label = [ones(1,50) 2*ones(1, 50)];

% test sample
y1 = randn(d,1); % observation 1
y2 = randn(d,1); % observation 2

y = [y1; y2];


%% Parameter Settings
options.diffnorm = 1e-3;
options.nbitermax = 15;
options.lambdareg = 5e-2;
s = 5;



[C]=JDSR(X,y,s, task_ind, label, options);

figure
stem(C, 'linewidth',2)
hold on
label_scaled = label/2*max(C(:));
plot(label_scaled, 'r:', 'linewidth',2)

legend('Observation 1', 'Observation 2', 'Class')