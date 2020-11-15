function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n=length(theta);
for iter = 1:num_iters
	z=zeros(m,1);
	thetaT=zeros(n,1);
	z=X*theta;
	z=z-y;
	for i=1:n
		w=0;
		q=zeros(m,1);
		q=z.*X(:,i);
		w=sum(q);
		thetaT(i)=(alpha/m)*w;
	end
	theta=theta-thetaT;
	J_history(iter) = computeCost(X, y, theta);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, theta);

end

end
