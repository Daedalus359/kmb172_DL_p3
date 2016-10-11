%use tanh, scaled and offset for range 0 to 1
%also compute the gradient of activation fnc
function [y_squash,y_prime] = squash(x_vec)
[nrows,ncols] = size(x_vec);
  offset = ones(nrows,ncols);
  y_squash = 0.5*(tanh(x_vec)+offset);
  y_prime = 2*y_squash.*(1-y_squash);
