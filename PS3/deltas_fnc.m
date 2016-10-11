function [deltavec2, deltavec1] = deltas_fnc(Y_out,targets, Gprime1,Gprime2,W2)
deltavec2 = (Y_out-targets).*Gprime2; %this should be a scalar, since only one output node with a single bias term
deltavec1 = (W2'*deltavec2).*Gprime1; %this should have as many terms as biasvec2, inputs to featurevec	