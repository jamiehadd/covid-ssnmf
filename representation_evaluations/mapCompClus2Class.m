function [labels_perm, C_perm] = mapCompClus2Class(labels, comp_clus)
% With ground truth labels, we compute an "optimal" permutation of the
% computed clusters in comp_clus. We return what this permutation yields as
% labels_perm, along with the confusion matrix of that corresponding
% permutation C_perm. 
% Inputs:
%       labels : "true" class labels for data. 
%                   - n array with integers in {1,2,...,k} (k classes)
%       comp_clus : "computed" cluster labelings
%                   - n array with integers in {1,2,...,k} (k clusters)
% 
% Outputs:
%       labels_perm : "optimal" permutation of comp_clus labelings to have 
%           greatest accuracy when compared to true class labels
%       C_perm : confusion matrix for the associated optimal permutation
%               - (k,k) matrix, where rows represent breakdown of computed
%                       clusters and columns represent true class
%                       breakdowns.



k = length(unique(labels));
C = zeros(k,k);
for c=1:k
    row = [];
    for i=1:k
        row = [row, sum((labels == i).*comp_clus == c)];
    end
    
   C(c,:) =  row;
end

P = perms(1:k);
num_perms = size(P,1);
max_diag = 0;
max_ind = 1;
for i=1:num_perms
    Cp = [];
    for j=1:k
        Cp = [Cp; C(P(i,j),:)];
    end
   if trace(Cp) > max_diag
      max_diag = trace(Cp);
      max_ind = i;
   end
end

C_perm = [];
for i=1:k
    C_perm = [C_perm; C(P(max_ind, i),:)];
end

max_perm = P(max_ind,:);
labels_perm = zeros(1,length(labels));
for c=1:k
   labels_perm(find(comp_clus == max_perm(c))) =  c;
end

end

