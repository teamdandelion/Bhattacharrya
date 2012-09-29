function [ e_bhatta ] = empirical_bhatta( X1,X2,kernel,parameter,eta )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

[n1, ~] = size(X1);
[n2, ~] = size(X2);
n = n1+n2;
[~, Kuc, Kc] = kernel_matrix(X1,X2,kernel,parameter);
G = bhatta_gs(Kc, n1, n2);

mu1 = sum(Kuc(1:n1,:)  *G,1)/n1;
mu2 = sum(Kuc(n1+1:n,:)*G,1)/n2;

Eta = eye(n-2) * eta;
S1 = (G'*Kc(:,1:n1)  *Kc(1:n1,:)  *G)/n1 + Eta;
S2 = (G'*Kc(:,1+n1:n)*Kc(n1+1:n,:)*G)/n2 + Eta;


mu3 = (inv(S1) * mu1' + inv(S2) * mu2')'/2;
%S3w = 2 * inv(invS1 + invS2);
S3 = 2 * inv(inv(S1) + inv(S2));


det1 = det(S1)^(-1/4);
det2 = det(S2)^(-1/4);
det3 = det(S3)^(1/2);

exp1 = -mu1 * inv(S1) * mu1' /4;
exp2 = -mu2 * inv(S2) * mu2' /4;
exp3 =  mu3 * S3    * mu3' /2;

dett = det1 * det2 * det3;
expt = exp(exp1 + exp2 + exp3);

e_bhatta = dett * expt;
e_bhatta_components = [dett, expt, e_bhatta]

end

