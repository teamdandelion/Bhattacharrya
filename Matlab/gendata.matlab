function [ X1,X2 ] = gendata( n1, n2, d )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
X1 = randn(n1,d);
%X2 = randn(n2,d);
X2 = X1 + .001 * rand(n1,d);
end

