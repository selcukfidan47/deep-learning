function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% forward propagation to compute cost
% W1 is 25 * 64, b1 is 25 * 1, data is 64 * 10000
% y_val = [b1, W1] * [ones(1, size(data, 2)); data];
% y_val = sigmoid(y_val);
% y_val = [b2, W2] * [ones(1, size(data, 2)); y_val];
% y_val = sigmoid(y_val);

% size(data, 1) % 64
% size(W1)   % 25 64
% size(W2)   % 64 25
% size(b1)   % 25  1
% size(b2)   % 64  1

m = size(data, 2);

z_2 = W1 * data + repmat(b1, 1, m);
a_2 = sigmoid(z_2); % 25 10000

rho_hat = sum(a_2, 2) / m; % This doesn't contain an x because the data
                       % above "has" the x

z_3 = W2 * a_2 + repmat(b2, 1, m);
a_3 = sigmoid(z_3); % 64 10000

diff = a_3 - data;
sparse_penalty = kl(sparsityParam, rho_hat);
J_simple = sum(sum(diff.^2)) / (2*m);

reg = sum(W1(:).^2) + sum(W2(:).^2);

cost = J_simple + beta * sparse_penalty + lambda * reg / 2;

% Backpropogation
% f'(z) = a * (1-a)

% for i=1:m,
%   delta3 = -(data(:,i) - a_3(:,i)) .* prime(z_3(:,i)); 
%   delta2 = W2'*delta3 .* prime(z_2(:,i));
 
%   W2grad = W2grad + delta3*a_2(:,i)' / m;
%   W1grad = W1grad + delta2*a_1(:,i)' / m; 
%   b2grad = b2grad + delta3 / m;
%   b1grad = b1grad + delta2 / m;
% end;

delta_3 = diff .* (a_3 .* (1-a_3));   % 64 10000

d2_simple = W2' * delta_3;   % 25 10000
d2_pen = kl_delta(sparsityParam, rho_hat);

delta_2 = (d2_simple + beta * repmat(d2_pen,1, m)) .* a_2 .* (1-a_2);

b2grad = sum(delta_3, 2)/m;
b1grad = sum(delta_2, 2)/m;

W2grad = delta_3 * a_2'/m  + lambda * W2; % 25 64
W1grad = delta_2 * data'/m + lambda * W1; % 25 64


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

% compute KL divergence
function ans = kl(r, rh)
  ans = sum(r .* log(r ./ rh) + (1-r) .* log( (1-r) ./ (1-rh)));
end

% compute derivative w.r.t rh
function ans = kl_delta(r, rh)
  ans = -(r./rh) + (1-r) ./ (1-rh);
end

% compute derivative w.r.t x for sigmoid function
function pr = prime(x)
    pr = sigmoid(x) .* (1 - sigmoid(x));
end

