function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta1 = [0 , theta'(2:length(theta))];

J = ( ( ((-y)' * log(sigmoid(X*theta))) - ((1-y)' * log(1 - sigmoid(X*theta))) ) / m  ) + ( (lambda / (2*m)) * (theta1 * theta1') );

grad = ( ( X' * (sigmoid(X * theta) - y)) / m);

for i = 1:length(grad)
  if i == 1
    grad(i) = grad(i);
else
  grad(i) = grad(i) + ((lambda/m)*(theta(i)));
  end
endfor

% =============================================================
end