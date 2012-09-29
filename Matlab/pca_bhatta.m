function [ bhatta ] = pca_bhatta( X1, X2, kernel, parameter, eta, r )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[n1, ~] = size(X1);
[n2, ~] = size(X2);
n = n1+n2;
[~, Kuc, Kc] = kernel_matrix(X1,X2,kernel,parameter);
G = bhatta_gs(Kc, n1, n2);

Kc1 = Kc(1:n1, 1:n1);
Kc2 = Kc(n1+1:n, n1+1:n);

[alpha1, Lam1] = eigs(Kc1,r);
[alpha2, Lam2] = eigs(Kc2,r);
Lam1 = Lam1 / n1; 
Lam2 = Lam2 / n2;


%Beta1, Beta2 hold normalized eigenvector coefficients so that 
% <v_i | v_i> = 1. Note <v_i| = \sum_j beta(j,i) <x*_j|
% Note we also shift Beta2 down to the bottom portion of the matrix
% Because Beta2 is composed of vectors from set X2, i.e. |x*_i> for
% i in n1+1 : n
Beta1 = zeros(n,r);
Beta2 = zeros(n,r);

for i=1:r
    Beta1(1:n1  ,i) = alpha1(:,i) / (n1 * Lam1(i,i))^(.5);
    Beta2(n1+1:n,i) = alpha2(:,i) / (n2 * Lam2(i,i))^(.5);
end

Beta = [Beta1, Beta2];

mu1_e = sum(Kuc(1:n1, :) * Beta1, 1) / n1;
mu2_e = sum(Kuc(n1:n, :) * Beta2, 1) / n2;


Eta_e = eta * eye(r);
Eta_r = eta * eye(n);
S1_e = Lam1 + Eta_e;
S2_e = Lam2 + Eta_e;

d1 = prod(diag(Lam1) + eta) ^ -.25;
d2 = prod(diag(Lam2) + eta) ^ -.25;

e1 = exp(-mu1_e * inv(S1_e) * mu1_e' / 4);
e2 = exp(-mu2_e * inv(S2_e) * mu2_e' / 4);

[Q,R] = qr(Beta');
%Q : (n x 2r) R : (2r x 2r)
mu1_r = zeros(1,2*r);
mu2_r = zeros(1,2*r);
mu1_r(1,1:r) = mu1_e;
mu2_r(1,1:r) = mu2_e;
mu1_r = (R * mu1_r')';
mu2_r = (R * mu2_r')';

S1_r = zeros(n,n);
S2_r = zeros(n,n);
S1_r(1:r,1:r) = Lam1;
S2_r(r+1:2*r, r+1:2*r) = Lam2;
S1_r = R' * (S1_r + Eta_r) * R; % Figure it out
S2_r = R' * (S2_r + Eta_r) * R; % Figure it out too

mu3_r = .5 * (inv(S1_r) * mu1_r' + inv(S2_r) * mu2_r')';
S3 = 2 * inv(inv(S1_r) + inv(S2_r));

d3 = det(S3) ^ .5;
e3 = exp(mu3_r * S3 * mu3_r.T / 2);

%% PCA Bhatta - eigendecomposition
pca_S1 = eye(n-2) * z + (G'*Kc*Beta1 * lam1 * (G'*Kc*Beta1)');
pca_S2 = eye(n-2) * z + (G'*Kc*Beta2 * lam2 * (G'*Kc*Beta2)');

pca_S3 = 2 * inv( inv(pca_S1) + inv(pca_S2) );
pca_mu3 = .5 * (inv(pca_S1) * pca_mu1' + inv(pca_S2) * pca_mu2')';

raw_det1 = (prod(diag(lam1)+z) * z^(n-r))^-.25;% - To my surprise, using these determinants didn't work
raw_det2 = (prod(diag(lam2)+z) * z^(n-r))^-.25;
pca_det1 = det(pca_S1)^-.25;
pca_det2 = det(pca_S2)^-.25;
pca_det3 = det(pca_S3)^.5;

pca_exp1 = -(pca_mu1 * inv(pca_S1) * pca_mu1')/4;
pca_exp2 = -(pca_mu2 * inv(pca_S2) * pca_mu2')/4;
pca_exp3 = (pca_mu3  * pca_S3      * pca_mu3')/2;

pca_det = pca_det1*pca_det2*pca_det3;
pca_exp = exp(pca_exp1 + pca_exp2 + pca_exp3);

pca_bhatta = pca_exp * pca_det
pca_bhata_components = [pca_det, pca_exp, pca_bhatta];



end

