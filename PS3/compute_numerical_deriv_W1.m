function dW1_num = compute_numerical_deriv_W1(i,j,X0,W1,bias1_vec,W2,bias2,target_vals)
dw = 0.0001;
[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);
%compute error(s):
E1 = err_fnc(Y_out,target_vals)	
W1(i,j) = W1(i,j)+dw;
[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);
E2 = err_fnc(Y_out,target_vals)	
dW1_num = (E2-E1)/dw;