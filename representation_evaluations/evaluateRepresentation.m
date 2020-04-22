%% Script for running SC and kmeans clustering on the features we currently have
% 

clear all
clc

save_loc = 'clusteringResults/'; % CHANGE to be the folder location where you want to store the results

%% Prepare data (num_samples, dimension of rep) 
% Change the details here for the .mat files you want. Just need to make
% sure that X = (num_samples, dim) array and labels has labels
% corresponding to rows of X.

%% COVIDNET
% save_title = 'COVIDNet_1024';
% 
% load('representations_train_1024.mat');
% load('representations_test_1024.mat');
% load('train_labels.mat');
% load('test_labels.mat');
% 
% X = [representations_train_1024; representations_test_1024];
% labels = [train_labels, test_labels];


%% Densenet
save_title = 'Densenet_1024';
load('~/Dropbox/299J-COVID-19 X-ray Project/Code/cleanCOVID2_Densenet_1024.mat')
load('~/Dropbox/299J-COVID-19 X-ray Project/Code/cleanCOVID2.mat', 'labels');

X = features;
clear('features');



%% Prepare labels vector, start with class 1

if ~isempty(find(labels == 0))
    labels = labels + 1;
end

k = max(labels); % number of clusters we will seek
num_points = size(X, 1); % number of points in the dataset


%% Kmeans clustering, k classes - MATLAB function
[km_clus, ~] = kmeans(X, k);



%% Similarity Graph Construction, Weight Matrix W

% Hyperparameters for similarity graph construction
knn = 20;
Ltype = 'symmetric';
reweight = true;
sigma = 3.0;



% do knn nearest neighbors
dist = sqdist(X', X'); % if inefficient, can use k-nearest neighbor tree
[dist, index] = sort(dist, 2, 'ascend');
d_sp = dist(:, 2:knn+1);
j_sp = index(:, 2:knn+1);
clear dist index;

% compute the weights via the scaling by mean of closest knn dist.
% (Usually good to do)
if reweight
    dsum_sp = sum(d_sp, 2);
    dmean_sp = dsum_sp / knn;
else
    dmean_sp = ones(num_points, 1);
end
    
w_sp = bsxfun(@rdivide, d_sp, dmean_sp);
w_sp = exp(-(w_sp .* w_sp)/ sigma);

i_sp = reshape((1:num_points)' * ones(1, knn), 1, num_points * knn , 1);
j_sp = reshape(j_sp, numel(j_sp), 1);

% Create a sparse weight matrix, and make symmetric
W = sparse(i_sp, j_sp, w_sp, num_points, num_points);
W = .5 * (W + W');



%% Run MATLAB SC on W

[sc_clus, V] = spectralcluster(W, single(k), 'Distance', 'precomputed',...
                'LaplacianNormalization', Ltype);
   
            
    
            
%% Check confusion permutation, output confusion matrix

if size(labels) ~= size(sc_clus)
    sc_clus = sc_clus';
end

if size(labels) ~= size(km_clus)
    km_clus = km_clus';
end


[labels_sc, Cp_sc] = mapCompClus2Class(labels, sc_clus);
[labels_km, Cp_km] = mapCompClus2Class(labels, km_clus);


if size(labels) ~= size(labels_sc)
    labels = labels';
end
    
acc_sc = length(find(labels_sc == labels))/num_points;
acc_km = length(find(labels_km == labels))/num_points;


%% Plotting to visualize clusters (projected onto the first few eigenvectors found in Spectral Clustering
i1 = 1;
i2 = 2;

figure
subplot(1,3,1)
gscatter(V(:,i1), V(:,i2), sc_clus);
title('SC Clusters')
subplot(1,3,2)
gscatter(V(:,i1), V(:,i2), labels);
title('Ground Truth')
subplot(1,3,3)
gscatter(V(:,i1), V(:,i2), km_clus);
title('Kmeans onto SC projection')


%% Save data

save(strcat(save_loc, save_title,'_', string(knn),'_knn.mat'), ...
    'Cp_sc', 'labels_sc', 'acc_sc', 'Cp_km', 'labels_km', 'acc_km')