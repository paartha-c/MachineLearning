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