%% Computes Closed-Form Bhattacharrya Kernel, 
%% PCA Kernelized Bhattacharrya Kernel, and 
%% Empirical-Covariance Bhattacharrya Kernel
%% for random data sets.

% Data parameters
n1 = 100;
n2 = 100;
d = 2;

% Model Parameters
z = .1; % Normalization constant for PCA kernel
r = 17;  % Number of eigenvectors for PCA kernel
sigma = 2; % Sigma for gaussian RBF kernel
eta = .1; % Normalization constant for empirical kernel

results = zeros(8,3);

n_iterations = 2;

for iteration=1:n_iterations

D0 = randn(n1,d);
D1 = D0 + rand(n2,d) / 1000; % Verify identity K(X,X) = 1
D2 = randn(n2,d); % How does kernel perform iid data
D3 = rand(n2,d); % Uniform rather than normal distribution
D4 = randn(n2,d)*2+2; % Linear transformation
D5 = (randn(n2,d)+1).^3; %Non-linear transformation
D6 = (D0+1).^3; %non-linear transformation of the D0 dataset;
D7 = rand(n2,d)/100 + eye(n2,d); %Totally different data - should have low similarity
D8 = rand(n2,d)/100 + eye(n2,d)*5; % Scaled version of D7

rdata = {D1,D2,D3,D4,D5,D6,D7,D8};
X1 = D0;

for trial=1:8
    % Set-up
    X2 = rdata{1,trial};
    
    [n1, dummy] = size(X1);
    [n2, dummy] = size(X2);
    n = n1 + n2;
    X = [X1;X2];

    nk_mu1 = sum(X1)/n1; % results in a 1xd row vector
    nk_mu2 = sum(X2)/n2;
    nk_X1c = X1 - repmat(nk_mu1,n1,1);
    nk_X2c = X2 - repmat(nk_mu2,n2,1);

    nk_S1 = nk_X1c'*nk_X1c / n1;
    nk_S2 = nk_X2c'*nk_X2c / n2;

    nk_invS1 = inv(nk_S1);
    nk_invS2 = inv(nk_S2);

    nk_mu3 = .5* (nk_invS1 * nk_mu1' + nk_invS2 * nk_mu2')';
    nk_S3 = 2 * inv(nk_invS1 + nk_invS2);

    nk_det = det(nk_S1)^(-1/4) * det(nk_S2)^(-1/4)*det(nk_S3)^(1/2);
    nk_a = -.25* nk_mu1 * nk_invS1 * nk_mu1';
    nk_b = -.25* nk_mu2 * nk_invS2 * nk_mu2';
    nk_c = .5*   nk_mu3 * nk_S3 * nk_mu3';
    nk_exp = exp(nk_a + nk_b + nk_c);
    nk_bhatta = nk_det * nk_exp;
    nk_components = [nk_det, nk_exp, nk_bhatta];


    % Compute general kernel bhattacharrya components
    % K - n x n matrix where K_ij = <x_i | x_j>
    K=zeros(n,n); %K(i,j) = <x_i | x_j>
    for i=1:n
        for j=1:i
            K(i,j) = exp(-norm(X(i,:)-X(j,:))^2 / (2*sigma^2));
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
    %Kc(i,j) = <X*i | X*j>
    % All of this verified with linear & poly kernel

    % Now we pivot to computing the kernelized bhattacharrya using a
    % GS-basis

    G = eye(n);
    G(:,n) = [];
    G(:,n1)= [];
    % We get rid of two dimensions because the centering process reduces
    % the dimensionality of the data
    for ell=1:n-2
        for i=1:ell-1
            GKG = G'*Kc*G;
            G(:,ell) = G(:,ell) - GKG(ell,i) * G(:,i);
        end
        GKG = G'*Kc*G;
        G(:,ell) = G(:,ell) / (GKG(ell,ell))^(.5);
    end

    mu1 = sum(Kuc(1:n1,:)  *G,1)/n1;
    mu2 = sum(Kuc(n1+1:n,:)*G,1)/n2;

    e_mu1 = mu1;
    e_mu2 = mu2;
    pca_mu1 = mu1;
    pca_mu2 = mu2;

    %% Empirical Bhatta - no PCA / eigendecomposition
    e_S1 = eye(n-2)*z + (G'*Kc(:,1:n1)  *Kc(1:n1,:)  *G)/n1;
    e_S2 = eye(n-2)*z + (G'*Kc(:,1+n1:n)*Kc(n1+1:n,:)*G)/n2;

    %e_invS1=inv(e_S1);
    %e_invS2=inv(e_S2);

    e_mu3 = (inv(e_S1) * e_mu1' + inv(e_S2) * e_mu2')'/2;
    %e_S3w = 2 * inv(e_invS1 + e_invS2);
    e_S3 = 2 * inv(inv(e_S1) + inv(e_S2));


    e_det1 = det(e_S1)^(-1/4);
    e_det2 = det(e_S2)^(-1/4);
    e_det3 = det(e_S3)^(1/2);

    e_exp1 = -e_mu1 * inv(e_S1) * e_mu1' /4;
    e_exp2 = -e_mu2 * inv(e_S2) * e_mu2' /4;
    e_exp3 =  e_mu3 * e_S3    * e_mu3' /2;

    e_det = e_det1 * e_det2 * e_det3;
    e_exp = exp(e_exp1 + e_exp2 + e_exp3);

    e_bhatta = e_det * e_exp;
    e_bhatta_components = [e_det, e_exp, e_bhatta];


    Kc1 = Kc(1:n1, 1:n1);
    Kc2 = Kc(n1+1:n, n1+1:n);

    % Eigendecomposition of centered kernel matrix - for PCA kernel
    [alpha1, lam1] = eigs(Kc1,r);
    [alpha2, lam2] = eigs(Kc2,r);
    lam1 = lam1 / n1; 
    lam2 = lam2 / n2;


    %Beta1, Beta2 hold normalized eigenvector coefficients so that 
    % <v_i | v_i> = 1. Note <v_i| = \sum_j beta(j,i) <x*_j|
    % Note we also shift Beta2 down to the bottom portion of the matrix
    % Because Beta2 is composed of vectors from set X2, i.e. |x*_i> for
    % i in n1+1 : n
    Beta1 = zeros(n,r);
    Beta2 = zeros(n,r);

    for i=1:r
        Beta1(1:n1  ,i) = alpha1(:,i) / (n1 * lam1(i,i))^(.5);
        Beta2(n1+1:n,i) = alpha2(:,i) / (n2 * lam2(i,i))^(.5);
    end

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

    pca_bhatta = pca_exp * pca_det;
    pca_bhata_components = [pca_det, pca_exp, pca_bhatta];

    bhatta_values = [nk_bhatta, e_bhatta, pca_bhatta];
    results(trial,:) = results(trial,:) + bhatta_values;
end

end

results = results/n_iterations

