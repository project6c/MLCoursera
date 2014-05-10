function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
J_inter = 0;
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y = eye(num_labels)(y,:);
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
J = sum(sum(-y.*log(a3) -(1-y).*log(1-a3)));
J = J/m;
%for j = 1:m 
%	J_inter = 0;
%	y1 = zeros(num_labels,1);
%	a1 = X(j,:); %getting row of x
%	y1(y(j)) = 1; % creating y vector by the label
%	z2 = Theta1*a1';
%	a2 = sigmoid(z2);
%	a2 = [1;a2];
%	z3 = Theta2*a2;
%	a3 = sigmoid(z3);
%%	J = J + (J_inter/m);
%end

%regularization terms
Theta1_sum = sum(sum(Theta1(:,2:end).*Theta1(:,2:end)));
Theta2_sum = sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));
Reg_term = (lambda/(2*m))*(Theta1_sum+Theta2_sum);
J = J + Reg_term;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vecto''''''''''''''''''''''''''''''''''''''''''''''''''''''''''r of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
D3 = a3-y;
D2 = (D3*Theta2);

D2 = D2(:,2:end);
size(D2);
D2 = D2.*sigmoidGradient(z2);
Theta1_grad = D2'*a1;
Theta1_grad = Theta1_grad/m;
Theta2_grad = D3'*a2;
Theta2_grad = Theta2_grad/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
regTheta1_grad = (lambda/m)*Theta1;
regTheta1_grad(:,1) = 0;
Theta1_grad = Theta1_grad +regTheta1_grad;
regTheta2_grad = (lambda/m)*Theta2;
regTheta2_grad(:,1) = 0;
Theta2_grad = Theta2_grad +regTheta2_grad;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
