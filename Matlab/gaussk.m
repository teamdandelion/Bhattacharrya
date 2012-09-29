function [ k ] = gaussk( X,Y,sigma )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    
k = exp(-norm(X-Y)^2 / (2*sigma^2));

end

