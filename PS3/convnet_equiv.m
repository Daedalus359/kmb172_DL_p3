%do same as convnet, but use Wc as equivalent convolution,
%and input image SOH,
%and bias_array consistent-dim vectorize

%this convnet consists of:
%x0 = input image; 
%2-D convolution w/ "kernel" --> featuremap
% x_1 = tanh(featuremap + bias_array)
%u_2 = W*featurevec + y_bias
%y = g(u_2)
%[Y_out,X1,gprime1_vec,gprime2] = convnet_equiv(X0,W1,b1,W2,b2)

%assume X0 will have one or more images as column(s), and bias_vec is a column vec
function [Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias_vec1,W2,bias2)
%size_in_image = size(in_image_SOH)
%size_W_conv = size(W_kernel_equiv)
%size_bias_vec1 = size(bias_vec1)
%size_W2 = size(W2)

%say images are in column-vec form; 
[NPIX,n_images] = size(X0);
[x1_dim,x0_dim]=size(W1);
%expand bias vec to offset all images:
Bias_vecs = bias_vec1*ones(1,n_images);
bias_vecs_size = size(Bias_vecs);
 %feature_vec is same as u1
  U1 = W1*X0 + Bias_vecs; %if have multiple images as column vecs, will get multiple uvecs as column vecs
  U1_mean = mean(mean(U1))
  [X1,Gprime1] = squash(U1);

 U2 = W2*(X1) +bias2*ones(1,n_images);%net potential
 U2_mean = mean(U2)
 %size_u2 = size(u2)
 [Y_out,Gprime2] = squash(U2);
