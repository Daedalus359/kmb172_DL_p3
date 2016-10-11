function E = err_fnc(Y_out,target_vals)
E = 0.5*norm(Y_out-target_vals)^2;