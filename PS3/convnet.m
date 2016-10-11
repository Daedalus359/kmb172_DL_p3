%this convnet consists of:
%x0 = input image; 
%2-D convolution w/ "kernel" --> featuremap
% x_1 = squash(featuremap + bias)
%u_2 = W*featurevec + y_bias
%y = g(u_2)
function [y_squashed,x1,gprime1_vec,gprime2] = convnet(in_image,kernel,bias1_vec,W2,y_bias)
	
  featuremap = conv2(in_image,kernel,'valid');
%figure(2)
%imshow(featuremap/110) %scale intensities to max=1.0
%title('feature map')
[fm_rows,fm_cols] = size(featuremap);
npix_featuremap = fm_rows*fm_cols;
%bias_array = -ones(fm_rows,fm_cols)*10;
%featuremap_squashed = 0.5*tanh(featuremap+bias_array)+0.5;

%make a row vector of the featuremap, strung-out horizontally:
featurevec = reshape(featuremap',1,npix_featuremap);
%convert to  column vecs for u1, x1
u1 = featurevec'+bias1_vec; %add bias vec
[x1,gprime1_vec] = squash(u1); %apply activation fnc
 u_out = W2*(x1) +y_bias %net potential
  [y_squashed,gprime2] = squash(u_out);
