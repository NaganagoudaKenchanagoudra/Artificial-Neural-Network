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
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

eye_matrix=eye(num_labels);
y_matrix=eye_matrix(y,:);
X=[ones(m,1) X];
a1=X;
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);
cost=(1./m).*(sum(sum(-y_matrix.*log(a3)))-(sum(sum((1-y_matrix).*log(1-a3)))));
J=cost;
temp1=Theta1;
temp2=Theta2;
temp1(:,1)=0;
temp2(:,1)=0;
reg=(lambda./(2*m)).*(sum(sum(temp1.^2)) +sum(sum(temp2.^2)));
J=J+reg;

delta3=(a3-y_matrix);
z2g=sigmoidGradient(z2);
temp=Theta2(:,2:end);
delta2=(delta3*temp).*z2g;

Delta1=(delta2'*a1);
Delta2=(delta3'*a2);
grad1=Delta1./m;
grad2=Delta2./m;

temp1=Theta1;
temp2=Theta2;
temp1(:,1)=0;
temp2(:,1)=0;
reg=(lambda./m).*(sum(temp2));
Theta2_grad=grad2+reg;
Theta1_grad=grad1;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%        %
% Part 3: Implement regularization with the cost function and gradients.
%
%        % -------------------------------------------------------------
% =========================================================================
% Unroll gradients


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
