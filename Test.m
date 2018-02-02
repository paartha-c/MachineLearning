data = load('ex1data.txt');

y = data(:,2);
m=length(y);

X = [ones(m,1),data(:,1)];
theta = zeros(2,1);
alpha = 0.01;
iterations = 1500;

J = computeCost(X,y,theta)
theta = gradientDescent(X,y,theta,alpha,iterations);
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));
% Predict the price for new test data using identified values

predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n');

theta0_vals = linspace(-10,10,100);
theta1_vals = linspace(-1, 4, 100);

%initialize J_values with zeros

J_vals = zeros(length(theta0_vals),length(theta1_vals));

for i=1:length(theta0_vals)
  for j = 1:length(theta1_vals)
    t=[theta0_vals(i);theta1_vals(j)];
    J_vals(i,j)=computeCost(X,y,t);
  end
end

J_vals = J_vals';
size(J_vals)
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

function J = computeCost(X,y,theta)
  m = length(y);
  squareError = sum(((X * theta)- y).^2);
  J = 1/(2*m)*squareError;
  end
  
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  
  for iter = 1:num_iters
  h = X*theta;
  
  val0=sum((h-y) .* X(:,1))*(1/m);
  val1=sum((h-y) .* X(:,2))*(1/m);
  theta(1) = theta(1) - (alpha * val0);
  theta(2) = theta(2) - (alpha * val1);
  
  J_history(iter) = computeCost(X, y, theta);
  
  end
end