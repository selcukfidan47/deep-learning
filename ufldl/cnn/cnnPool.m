function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

% pool for every image
for im = 1:numImages
    % pool for every feature
    for ft = 1:numFeatures
        % pool for every row
        for r = 1:size(pooledFeatures, 3)
            % pool for every column
            for c = 1:size(pooledFeatures, 4)
                rs = (r - 1) * poolDim + 1; rt = rs + poolDim - 1;
                cs = (c - 1) * poolDim + 1; ct = cs + poolDim - 1;
                assert(rt <= convolvedDim, 'out of bound error for row end index');
                assert(ct <= convolvedDim, 'out of bound error for column end index');
                pooledFeatures(ft, im, r, c) = mean(mean(convolvedFeatures(ft, im, rs:rt, cs:ct)));
            end
        end
    end
end

end

