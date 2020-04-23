% Thomas Merkh, tmerkh@g.ucla.edu, April 2020, k-means method for clustering.
% This script performs k-means clustering multiple times and measures the purity each time.

% This script was written specifically for clustering the representations learned by DenseNet on Xray data.  Contact me to request the data set.
% The average purity of each class are held by the variables covid_avg, bacterial_avg, normal_avg, and viral_avg.  
% The centroids for an given kmeans instance are contained in the columns of variable 'mu'
% The labels for any given instance are contained by in the variable 'label'
% The true labels are called 'labels'

clear; % The PCA algorithms use "end" indexing so this is necessary

load cleanCOVID2_Densenet_1024.mat  % This creates labels and features, (N x 1) and (N x 1024) 
data = features';
labels = labels';

% Labels:
% 1 : covid
% 2 : bacterial
% 3 : normal
% 4 : viral

PCAon = false;

k = 4;                                                          % Number of labels
init = 0;                                                       % Bool for initialization
normal_avg = 0; covid_avg = 0; viral_avg = 0; bacterial_avg = 0;% Purities averaged over n_runs times performing K means
n_runs = 100;                                                   % Rerun the initialization and EM algorithm this many times to see on average the purity obtained. 
collisions = 0;                                                 % Number of k-means instances with labelling collisions, not used for computing purity average.
n_success = 0;                                                  % Number of k-means runs without any labelling collisions (used for correct averaging)

for q = 1:n_runs
  %%%%%%%%%%%%%%%%%%%%% K-means algorithm %%%%%%%%%%%%%%%%%%%%%
  n = size(data,2);                                                  % Number of points
  last = 0;
  if(init == 1)
    label = ceil(k*rand(1,n));                                       % Random partition initialization
  else
    mu = data(:,randi(n,1,3));                                       % Forgy initialization
    [~,label] = max(bsxfun(@minus,mu'*data,dot(mu,mu,1)'/2),[],1);   
  endif 

  % EM algorithm
  while any(label ~= last)
      E = sparse(1:n,label,1,n,k,n);                                 % transform label into indicator matrix
      mu = data*(E*spdiags(1./sum(E,1)',0,k,k));                     % compute mu of each cluster
      last = label;
      [~,label] = max(bsxfun(@minus,mu'*data,dot(mu,mu,1)'/2),[],1); % assign samples to the nearest centers
  endwhile
   
  % compute purities
  [purity_covid,v1]      = max([sum((labels == 1).*label == 1),sum((labels == 1).*label == 2),sum((labels == 1).*label == 3),sum((labels == 1).*label == 4)]);
  [purity_bacterial, v2] = max([sum((labels == 2).*label == 1),sum((labels == 2).*label == 2),sum((labels == 2).*label == 3),sum((labels == 2).*label == 4)]);
  [purity_normal, v3]    = max([sum((labels == 3).*label == 1),sum((labels == 3).*label == 2),sum((labels == 3).*label == 3),sum((labels == 3).*label == 4)]);
  [purity_viral, v4]     = max([sum((labels == 4).*label == 1),sum((labels == 4).*label == 2),sum((labels == 4).*label == 3),sum((labels == 4).*label == 4)]);
  
  purity_covid = purity_covid/sum(labels == 1);
  purity_bacterial = purity_bacterial/sum(labels == 2);
  purity_normal = purity_normal/sum(labels == 3);
  purity_viral = purity_viral/sum(labels == 4);
    
  % Don't store the purities for all the runs and then average, instead average along the way.
  % Only do this if all of the clusters are assigned unique labels.
  
  if( (v1 != (v2 || v3)) && (v1 != v4) && ((v2 != (v3 || v4)) && v3 != v4) )
      n_success = n_success + 1;
      normal_avg = normal_avg + (1.0/n_success)*(purity_normal - normal_avg); 
      covid_avg = covid_avg + (1.0/n_success)*(purity_covid - covid_avg); 
      viral_avg = viral_avg + (1.0/n_success)*(purity_viral - viral_avg); 
      bacterial_avg = bacterial_avg + (1.0/n_success)*(purity_bacterial - bacterial_avg);  
  else
      collisions = collisions + 1;
  endif
endfor


if(collisions > 0)
  disp('The number of collisions was:')
  collisions
endif

disp('The average purities found were:')
disp('Normal:')
disp(normal_avg)
disp('Covid:')
disp(covid_avg)
disp('Bacterial:')
disp(bacterial_avg)
disp('Viral:')
disp(viral_avg)


if(!PCAon)
  disp('Program Finished')
  return
endif

%% PCA - Project the data onto a 3D subspace for plotting.
[D N] = size(data);       % number of data points and dimension
M = 3;                    % Dimension of projection
p_coeffs = zeros(N,M);    % Projection coefficients
samplemean = mean(data)'; % Mean of all the data points
S = cov(data);            % Covariance 
S = (S + S')./2;          % To fix for roundoff errors
[Evecs, eigenvalues] = eig(S);
Evecs = real(Evecs); eigenvalues = real(eigenvalues); % Numerical precision isn't perfect
eigenvalues = diag(eigenvalues);
% Project onto first principle components 
P3 = Evecs(3,:)';
P2 = Evecs(2,:)';
P1 = Evecs(1,:)';

P3 = P3./norm(P3);
P2 = P2./norm(P2);
P1 = P1./norm(P1);
for i = 1:N
  p_coeffs(i,:) = [(data(i,:)' - samplemean)'*P1, (data(i,:)' - samplemean)'*P2, (data(i,:)' - samplemean)'*P3];
end

%% Associate each class with a color!
for i = 1:size(p_coeffs,1)
  if(labels(i) == 3)
    normal_class(end+1,1:M) = p_coeffs(i,:);
  elseif(labels(i) == 1)
    covid_class(end+1,1:M) = p_coeffs(i,:);
  elseif(labels(i)== 2)
    bacterial_class(end + 1,1:M) = p_coeffs(i,:);
  elseif(labels(i)== 4)
    viral_class(end + 1,1:M) = p_coeffs(i,:);
  endif
endfor

figure
hold on

norma = scatter3(normal_class(:,1), normal_class(:,2),normal_class(:,3),'b','MarkerFaceColor' , 'b');
covida = scatter3(covid_class(:,1), covid_class(:,2),covid_class(:,3),'r', 'MarkerFaceColor' , 'r');
virala = scatter3(viral_class(:,1), viral_class(:,2),viral_class(:,3),'k', 'MarkerFaceColor' , 'k');
bacta = scatter3(bacterial_class(:,1), bacterial_class(:,2),bacterial_class(:,3),'k', 'MarkerFaceColor' , 'y');


hTitle = title('3D Projection via PCA of the data points');
hXLabel = xlabel('Principle Component 1');
hYLabel = ylabel('Principle Component 2');
hZLabel = zlabel('Principle Component 3');
hLegend = legend([norma, covida, virala, bacta], 'Normal','COVID', 'Viral', 'Bacterial', 'Location', 'NorthWest');

set(gca, 'FontName', 'Helvetica')
set([hTitle, hXLabel, hYLabel, hZLabel], 'FontName', 'AvantGarde')
set([hLegend, gca], 'FontSize', 10)
set([hXLabel, hYLabel, hZLabel], 'FontSize', 10, 'FontWeight' , 'bold')
set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')

%xticks([location1 location2])
%xticklabels({'string1','string2'})
%xlim([0 1])

set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', 'XGrid', 'on', ...
    'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], 'LineWidth', 1)
hold off
