function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  
    %theta = theta - ((alpha * (((((X/ * theta) - y)))' * X) / length(y)))'
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    
      theta = theta - (((X * theta) - y )' * X) * alpha / length(y);
      J_history(iter) = (sum(((X*  theta) - y).^2))/(2*m);

    % ============================================================

    % Save the cost J in every iteration         
 

end
