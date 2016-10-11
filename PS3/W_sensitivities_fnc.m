function [W2sense,W1sense] = W_sensitivities_fnc(X0, X1, deltavec1,deltavec2)
W2sense = deltavec2*X1';
W1sense= deltavec1*X0';