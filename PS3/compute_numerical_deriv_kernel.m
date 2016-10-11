function dEdK_num = compute_numerical_deriv_kernel(in_image,Kernel,dKernel_map,bias1_vec,W2,bias2,target_val)
dk = 0.0001;
[y_out,x1,gprime1_vec,gprime2] = convnet(in_image,Kernel,bias1_vec,W2,bias2);
%compute error(s):
E1 = err_fnc(y_out,target_val)	
Kernel2 = Kernel+dKernel_map*dk
[y_out,x1,gprime1_vec,gprime2] = convnet(in_image,Kernel2,bias1_vec,W2,bias2);
E2 = err_fnc(y_out,target_val)
dEdK_num = (E2-E1)/dk;