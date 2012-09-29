function [ G ] = bhatta_gs( Kc, n1, n2 )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
n = n1+n2;
G = eye(n);
G(:,n) = [];
G(:,n1)= [];
% We get rid of two dimensions because the centering process reduces
% the dimensionality of the data
for ell=1:n-2
    for i=1:ell-1
        GKG = G'*Kc*G;
        G(:,ell) = G(:,ell) - GKG(ell,i) / GKG(i,i) * G(:,i);
    end
    GKG = G'*Kc*G;
    G(:,ell) = G(:,ell) / (GKG(ell,ell))^(.5);
end

end

