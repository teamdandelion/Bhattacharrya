function [ K, Kuc, Kc ] = kernel_matrix( X1, X2, kernel, parameter )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
X = [X1; X2];
[n1, dummy] = size(X1);
[n2, dummy] = size(X2);
n = n1+n2;

% Compute general kernel bhattacharrya components
% K - n x n matrix where K_ij = <x_i | x_j>
K=zeros(n,n); %K(i,j) = <x_i | x_j>
for i=1:n
    for j=1:i
        K(i,j) = kernel(X(i,:),X(j,:),parameter);
        % Polynomial kernel - corresponds to a known RKHS
        K(j,i) = K(i,j);
    end;
end;

%U1_i = <mu_1 | xi>

U1 = sum(K(1:n1,:),1)/n1;
U2 = sum(K(n1+1:n,:),1)/n2;
U=[repmat(U1,n1,1);repmat(U2,n2,1)];

m1m1 = sum(sum(K(1:n1,   1:n1  ))) / (n1*n1);
m1m2 = sum(sum(K(1:n1,   n1+1:n))) / (n1*n2);
m2m2 = sum(sum(K(n1+1:n, n1+1:n))) / (n2*n2);

mumu = zeros(n,n);
mumu(1:n1,   1:n1  ) = m1m1;
mumu(1:n1,   n1+1:n) = m1m2;
mumu(n1+1:n, 1:n1  ) = m1m2;
mumu(n1+1:n, n1+1:n) = m2m2;

Kcu = K - U; % <Centered | Uncentered>
%Kcu(i,j) = <x*i | xj>
Kuc = Kcu';
%Kuc(i,j) = <xi | x*j> = <x*j | xi>
%Kcu(i,j) = <X*i | Xj>
Kc = K - U - U' + mumu; % <Centered | Centered>


end

