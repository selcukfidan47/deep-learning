function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

pred = theta * data;
% subtract the maximum value of each column
pred = bsxfun(@minus, pred, max(pred, [], 1));
pred = exp(pred);
% normalization to a valid probability distribution
pred = bsxfun(@rdivide, pred, sum(pred));

% for i = 1:numCases
%     cost = cost + groundTruth(:, i) .* log(pred(:, i));
% end

cost = cost + sum(sum(groundTruth .* log(pred)));
cost = cost * (-1.0 / numCases) + (lambda / 2) * sum(sum(theta .* theta));

% compute gradient
% for i = 1:numClasses
%     diff = groundTruth(i, :) - pred(i, :);
%     diff = repmat(diff, inputSize, 1);
%     total = sum(data .* diff, 2)';
%     thetagrad(i, :) = (-1.0 / numCases) * total + lambda * theta(i, :);
% end

% vectorized implementation can significantly speed up the training phase
thetagrad = lambda * theta + (-1.0 / numCases) * ((groundTruth - pred) * data');

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

