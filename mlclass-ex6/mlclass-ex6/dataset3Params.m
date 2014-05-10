function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C = 0.1;
Cret = 0;
sigmaret = 0;
sigma = 0.03;
error = 0;
newSigma = sigma;
newC = C;
minerror = 999990.0;
for i = 1:10
	for j = 1:10
		model=svmTrain(X, y, newC, @(x1, x2) gaussianKernel(x1, x2, newSigma));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		if error <= minerror
			modelfinal = model;
			error
			minerror = error
			Cret = newC;
			sigmaret = newSigma;
		endif
		newSigma = sigma*j*5;
	end
	newSigma;
	newC = C*i*10
	minerror
end

C = Cret
sigma = sigmaret




% =========================================================================

end
