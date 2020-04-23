function [D] = sqdist(X, Y)
%--------------------------------------------------------------------------
% Author: Xiyang Luo <xylmath@gmail.com> , UCLA 
%
% This file is part of the diffuse-interface graph algorithm code. 
% There are currently no licenses. 
%
%--------------------------------------------------------------------------
% Description: Computes pairwise square distance
%
% Usage: 
%       X = (x1,x2,\dots, xm) m col vectors
%       Y = (y1,y2,\dots, yn) n col vectors
%       D_ij = |xi-yj|^2 
%--------------------------------------------------------------------------

m = size(X,2);
n = size(Y,2);
Yt = Y';  
XX = sum(X.*X,1);        
YY = sum(Yt.*Yt,2);      
D = XX(ones(1,n),:) + YY(:,ones(1,m)) - 2*Yt*X;
end

