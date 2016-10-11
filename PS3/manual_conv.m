%operate w/ hand-designed convolution kernel
clear all
load cnn_data
%test_image = images_w_feature(1,:,:)
[N_IMAGES,NROWS,NCOLS] = size(images_w_feature)
num_image_pixels = NROWS*NCOLS
%manually design a kernel:
mone = -ones(1,11);
lineseg = ones(1,11);
%kernel = [mone;lineseg;mone]
%a 3x11 kernel:
kernel = [lineseg*0;lineseg*10;lineseg*0] %try kernel = feature
%do a conv2, just to get dimensions:
 test_image = squeeze(images_w_feature(1,:,:)); 
featuremap = conv2(test_image,kernel,'valid');
[fm_rows,fm_cols] = size(featuremap);
npix_featuremap = fm_rows*fm_cols;


%equivalent linear weights for convolution, mapping image_vec_SOH onto feature_vec:
%function [W_conv,x1_dim,x2_dim]= W_conv_equiv(sample_image,kernel)

[W_conv,x1_dim,x2_dim] = W_conv_equiv(test_image,kernel);
%size(W_conv)
%test  if Octave will tolerate dimensions of kernel-injection maps:
% [W_conv2,x1_dim,x2_dim,W_conv_alt,Kvecs_map]=W_conv_equiv2(test_image,kernel); %works OK, but need to suppress print output


%choose some network values
%make a set of weights, W, w/ all equal weights, maps featuremap pixels (as vector) onto 
%scalar output, y
W2 = ones(1,npix_featuremap);
%bias values for featuremap; output = g(W*x+b)
bias_array = -ones(fm_rows,fm_cols)*100;
bias_vec1 = reshape(bias_array',1,fm_rows*fm_cols);
y_bias = -1;  %output bias
%try all images in the list
n_success_w_features=0;
DO_IMAGES = min([10,N_IMAGES])
for i_image=1:DO_IMAGES %N_IMAGES
  test_image = squeeze(images_w_feature(i_image,:,:)); 
  figure(1)
  imshow(test_image)
  title('test image')
 
 %convnet does forward simu of network
  [y_out ,featuremap_squashed]= convnet(test_image,kernel,bias_vec1',W2,y_bias);
  if (y_out>0.5) 
  	n_success_w_features = n_success_w_features+1
  	end
figure(3)
imshow(featuremap_squashed)
title('featuremap squashed')
%%% ALTERNATIVE: use equivalent W_conv:
test_image_SOH= reshape(test_image',1,num_image_pixels);
[y_squashed,featurevec_squashed] = convnet_equiv(test_image_SOH',W_conv,bias_vec1',W2,y_bias);
y_out_equiv = y_squashed
featuremap2 = reshape(featurevec_squashed',fm_cols,fm_rows)'; %???
%result is identical
figure(5)
imshow(featuremap2)
title('using W conv')


display('hit enter to continue')
pause
end
success_rate_w_features = n_success_w_features/DO_IMAGES
display('hit enter to continue')

pause
%now try all images without feature:
[N_IMAGES,NPIX,NCOLS] = size(images_wo_feature)
display('evaluating images without feature')
n_success_wo_features=0;
for i_image=1:DO_IMAGES %N_IMAGES
  test_image = squeeze(images_wo_feature(i_image,:,:)); 
  figure(1)
  imshow(test_image)
  title('test image')
 
 %convnet does forward simu of network
  [y_out ,featuremap_squashed]= convnet(test_image,kernel,bias_vec1',W2,y_bias);
  if (y_out<0.5)
  	n_success_wo_features=n_success_wo_features+1
  end
figure(3)
imshow(featuremap_squashed)
title('featuremap squashed')
test_image_SOH= reshape(test_image',1,num_image_pixels);
[y_squashed,featurevec_squashed] = convnet_equiv(test_image_SOH',W_conv,bias_vec1',W2,y_bias);
y_out_equiv = y_squashed
featuremap2 = reshape(featurevec_squashed',fm_cols,fm_rows)'; %???
figure(5)
imshow(featuremap2)
title('using W conv')


display('hit enter to continue')
pause
end
success_rate_wo_features = n_success_wo_features/DO_IMAGES
