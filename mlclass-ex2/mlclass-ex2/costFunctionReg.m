function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
size_theta_red = size(theta,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

constant_factor = lambda/(2*m);
constant2 = (-1/m);
hypothesis = sigmoid(X*theta);
J = constant2*sum(log(hypothesis).*y + log(1-hypothesis).*(1-y));
theta_prod = theta(2:end)'*theta(2:end);
%theta_prod(1) = 0;
J = J + constant_factor*theta_prod;


grad = X'*(hypothesis-y);
temp = theta;
temp(1) = 0;
temp = lambda*temp;
grad = grad + temp;

%grad(1) = (hypothesis-y)'*X(:,1);
%size_X_red = size(X,2);
%size_grad_red = size(grad);
%X_inter = X(:,2:size_X_red);
%grad_inter = grad(2:size_grad_red);
%sum1 = X_inter'*(hypothesis-y);
%sum2 = lambda*grad_inter;
%total_sum = sum1 + sum2;

%grad(2:size(grad)) = total_sum;
grad = (1/m)*grad;

% =============================================================

end
