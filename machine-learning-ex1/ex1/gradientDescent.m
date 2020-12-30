function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples

for iter = 1:num_iters
  
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
  theta = theta - (((X * theta) - y )' * X)' * alpha / length(y);
  %J_history(iter) = (sum(((X*  theta) - y).^2))/(2*m);
end

    % ============================================================

    % Save the cost J in every iteration    
    %