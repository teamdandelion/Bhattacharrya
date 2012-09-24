% Non-kernelized Bhattacharrya
nk_mu1 = sum(X1)/n1; % results in a 1xd row vector
nk_mu2 = sum(X2)/n2;
nk_X1c = X1 - repmat(nk_mu1,n1,1);
nk_X2c = X2 - repmat(nk_mu2,n2,1);

nk_S1 = nk_X1c'*nk_X1c / n1;
nk_S2 = nk_X2c'*nk_X2c / n2;

%nk_invS1 = inv(nk_S1);
%nk_invS2 = inv(nk_S2);

nk_mu3 = .5* (inv(nk_S1) * nk_mu1' + inv(nk_S2) * nk_mu2')';
nk_S3 = 2 * inv(inv(nk_S1) + inv(nk_S2));

nk_det = det(nk_S1)^(-1/4) * det(nk_S2)^(-1/4)*det(nk_S3)^(1/2);
nk_a = -.25* nk_mu1 * inv(nk_S1) * nk_mu1';
nk_b = -.25* nk_mu2 * inv(nk_S2) * nk_mu2';
nk_c = .5*   nk_mu3 * nk_S3 * nk_mu3';
nk_exp = exp(nk_a + nk_b + nk_c);
nk_bhatta = nk_det * nk_exp;
