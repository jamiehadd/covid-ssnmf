% Thomas Merkh, tmerkh@g.ucla.edu, k-means method for clustering.
% This script performs k-means clustering multiple times and measures the purity each time.
% The average purity of each class are held by the variables normal_avg, covid_avg, pnuemonia_avg. 
% The centroids for an given kmeans instance are contained in the columns of variable 'mu'
% The labels for any given instance are contained by in the variable 'label'
% The true labels are called 'labels'

clear; % The PCA algorithms use "end" indexing so this is necessary
data = load('/media/tmerkh/G_Drive/representations_test_256');
labels = load('/media/tmerkh/G_Drive/test_labels');

PCAon = false;

data = data.representations_test_256';                 % labels is (1 by n), with 3 classes, normal == 0, COVID-19 == 1, and pneumonia == 2.
labels = labels.test_labels;                           % data is (D by n), where n = number of samples and D is the dimension of the representation. 
labels = labels + 1;                                   % labels are 1,2,3 instead of 0,1,2.

k = 3;                                                 % Number of labels
init = 1;                                              % Bool for initialization
normal_avg = 0; covid_avg = 0; pneumonia_avg = 0;      % Purities averaged over n_runs times performing K means
n_runs = 1000;                                         % Rerun the initialization and EM algorithm this many times to see on average the purity obtained. 
collisions = 0;                                        % Counts collisions of class labels, if any
n_success = 0;                                         % Counts number of runs without collisions (used for correct averaging since q can not be)

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
   
  % compute purities - as calculated here, there is a potential collision in that two clusters could be assigned the same class label.
  [purity_normal, normal_indx]      = max([sum((labels == 1).*label == 1),sum((labels == 1).*label == 2),sum((labels == 1).*label == 3)]);
  [purity_covid, covid_indx]        = max([sum((labels == 2).*label == 1),sum((labels == 2).*label == 2),sum((labels == 2).*label == 3)]);
  [purity_pneumonia, pneumonia_indx]= max([sum((labels == 3).*label == 1),sum((labels == 3).*label == 2),sum((labels == 3).*label == 3)]);
  
  purity_normal = purity_normal/sum(labels == 1);
  purity_covid = purity_covid/sum(labels == 2);
  purity_pneumonia = purity_pneumonia/sum(labels == 3);
  
  % class labels are held in: normal_indx, covid_indx, pneumonia_indx
  
  % Don't store the purities for all the runs and then average, instead average along the way.
  % Only do this if there wasn't a collision in class labels
  if((normal_indx != (pneumonia_indx || covid_indx)) && (covid_indx != pneumonia_indx))
    n_success = n_success + 1;
    normal_avg = normal_avg + (1.0/n_success)*(purity_normal - normal_avg); 
    covid_avg = covid_avg + (1.0/n_success)*(purity_covid - covid_avg); 
    pneumonia_avg = pneumonia_avg + (1.0/n_success)*(purity_pneumonia - pneumonia_avg);  
  else
    collisions = collisions + 1;
  endif
endfor

disp('The average purities found were:')
disp('Normal:')
disp(normal_avg)
disp('Covid:')
disp(covid_avg)
disp('Pneumonia:')
disp(pneumonia_avg)

if(collisions  > 0)
  disp('There were several collisions when labeling the clusters!')
  collisions
endif


if(!PCAon)
  disp('Program Finished')
  return
endif

%% PCA - Project the data onto a 3D subspace for plotting.
%% Unfortunately, PCA does not show much in the way of clustering.  
%% This doesn't invalidate the 
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
  if(labels(i) == 1)
    normal_class(end+1,1:M) = p_coeffs(i,:);
  elseif(labels(i) == 2)
    covid_class(end+1,1:M) = p_coeffs(i,:);
  else
    pneumonia_class(end + 1,1:M) = p_coeffs(i,:);
  endif
endfor

figure
hold on
norma = scatter3(normal_class(:,1), normal_class(:,2),normal_class(:,3),'b','MarkerFaceColor' , 'b');
covida = scatter3(covid_class(:,1), covid_class(:,2),covid_class(:,3),'r', 'MarkerFaceColor' , 'r');
pneu = scatter3(pneumonia_class(:,1), pneumonia_class(:,2),pneumonia_class(:,3),'k', 'MarkerFaceColor' , 'k');


hTitle = title('3D Projection via PCA of the data points');
hXLabel = xlabel('Principle Component 1');
hYLabel = ylabel('Principle Component 2');
hZLabel = zlabel('Principle Component 3');
hLegend = legend([norma, covida, pneu], 'Normal','COVID', 'Pneumonia', 'Location', 'NorthWest');

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
