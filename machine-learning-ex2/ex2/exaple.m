%1. The below code would load the data present in your desktop to the octave memory 
data = load('ex2data1.txt');
x = data(:,1:2);
y=data(:,3);

%2. Now we want to add a column x0 with all the rows as value 1 into the matrix.
%First take the length
m=length(y);
x=[ones(m,1),x];

g=inline('1.0 ./ (1.0 + exp(-z))');

theta = zeros(size(x(1,:)))';   % the theta has to be a 3*1 matrix so that it can multiply by x that is m*3 matrix
%  Now we calculate the hx or hypothetis, It is calculated here inside no. of iteration because the hupothesis has to be calculated for new theta for every iteration
z=x*theta;
h=g(z);     % Here the effect of inline function we used earlier will reflect
j=(1/m)*(-y'* log(h) - (1 - y)'*log(1-h))  % This formula is the vectorized form of the cost function J(theta) This calculates the cost function
grad=(1/m) *  x' * (h-y)     % This formula is the gradient descent formula that calculates the theta value.  
j