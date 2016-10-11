%convnet_backprop:
%iterate on network params, INCLUDING kernel params, to improve classification of sample data
clear all
load cnn_data

 eta_factor = 2 %multiply/divide eta by this factor w/ iterations;  tunable
 
 %choose dimensions of kernel:
 Krows=1 %3
Kcols= 11% 11

[N_IMAGES,NROWS,NCOLS] = size(images_w_feature)
targets_w_feature = ones(N_IMAGES,1); %col vec of corresponding desired y_out values for images w/ feature
[N_IMAGES_WO_FEATURES,NROWS,NCOLS] = size(images_w_feature)
targets_wo_feature = zeros(N_IMAGES_WO_FEATURES,1); %col vec of corresponding desired y_out values for images w/o feature

num_image_pixels = NROWS*NCOLS

%randomly initialize the Kernel (and equivalent flipped overlay)
K_overlay = rand(Krows,Kcols)
%note difference: K is an overlay mask, but corresponding conv kernel is flipped right/left AND upside down
Kernel= flipud(fliplr(K_overlay)) %kernel is related to overlay



%do a conv2, just to get dimensions:
 test_image = squeeze(images_w_feature(1,:,:)); 
 test_image_SOH= reshape(test_image',1,num_image_pixels);
featuremap = conv2(test_image,Kernel,'valid');
%for the "valid" range of convolution, featuremap will have smaller dimensions than input images
[fm_rows,fm_cols] = size(featuremap);
npix_featuremap = fm_rows*fm_cols;
%randomly initialize biases and layer-2 weights:
%DO need to be careful of ranges of values to assign
bias1_vec = 2*rand(npix_featuremap,1)-ones(npix_featuremap,1);
W2 = 2*rand(1,npix_featuremap)-ones(1,npix_featuremap);
bias2 = 2*rand(1,1)-1

%compute mapping for equiv matrix W_conv:
display('computing Kvecs_map; this could be slow...')
 [W_conv,x1_dim,x2_dim,W_conv_alt,Kvecs_map]= W_conv_equiv2(test_image,K_overlay);
 W1 = W_conv;
 [W1rows,W1cols]=size(W1)
 %repackage all images as column vecs:
 display('repackaging images as column vectors')
 all_images_as_cvecs = [];
 for i=1:N_IMAGES
    test_image = squeeze(images_w_feature(i,:,:)); 
    test_image_SOH= reshape(test_image',1,num_image_pixels);	
   image_vec = test_image_SOH';
   all_images_as_cvecs=[all_images_as_cvecs,image_vec];
 end	
  for i=1:N_IMAGES_WO_FEATURES
    test_image = squeeze(images_wo_feature(i,:,:)); 
    test_image_SOH= reshape(test_image',1,num_image_pixels);	
   image_vec = test_image_SOH';
   all_images_as_cvecs=[all_images_as_cvecs,image_vec];
 end	
 X0 = all_images_as_cvecs;
 %%%%%%%%%%%%%%%DEBUG
 % X0 = all_images_as_cvecs(:,1); %DEBUG!!! try single image
 [NPIX,N_ALL_IMAGES] =   size(X0)
 ones_cvec_nimages = ones(N_ALL_IMAGES,1);
 %images are now all column vecs; create corresponding targets in corresponding columns:
 target_vals = ones(1,N_IMAGES);
 target_vals = [target_vals,zeros(1,N_IMAGES_WO_FEATURES)];
 %%%%%%%%%%DEBUG
 %target_vals = [1]
 
 %that's all the preliminaries; simulate and train networks
%here is simulation of network applied to all training images:
[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);
%scale W2 to get reasonable U2 values:
 U2_mean = mean(W2*(X1));
 W2 = 0.5*W2/abs(U2_mean); %aim for U2 around 0.5

[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);


%compute error(s): compare network outputs to target values
E_new = err_fnc(Y_out,target_vals)
%compute bias sensitivities: delta vectors, per theory
[deltasvec2, deltasvec1] = deltas_fnc(Y_out,target_vals, Gprime1,Gprime2,W2);
%batch mode: add all of these together:
deltavec2 = deltasvec2*ones_cvec_nimages;
deltavec1 = deltasvec1*ones_cvec_nimages;

%compute sensitivities w/rt all weights, treating the convolution as an equivalent fully-connected layer
[W2sense,W1sense] = W_sensitivities_fnc(X0, X1, deltasvec1,deltasvec2);

deltasvec2_size = size(deltasvec2)
deltasvec1_size = size(deltasvec1)
W2sense_size = size(W2sense)
W1sense_size = size(W1sense)

%now, use derivs to update params w/ backprop
eta = 0.000001; %init value of eta
dE_expected=0;


%display('ready to do backprop iterations; hit a key')
%kb = kbhit(); %manual breakpoint
E_hist = [];
E_new = E_new*2; %start off w/ large value to coerce first iteration 
N_ITERS=20000
n_iter=0
while n_iter<N_ITERS
  n_iter=n_iter+1
  if (!(mod(n_iter,100)))
  	figure(1)
  	plot(E_hist)
  	Kernel
  	save cnn_params bias2 W2 bias1_vec Kernel
  	sleep(1)
  	end

%note difference: K is an overlay mask, but corresponding conv kernel is flipped right/left AND upside down
K_overlay =  flipud(fliplr(Kernel)); %overlay is rotated Kernel
%equiv W1 from current values of Kernel/Overlay:
k_overlay_SOH = reshape(K_overlay',1,Krows*Kcols);
W_conv_alt_SOH = k_overlay_SOH*Kvecs_map; %compute W1 weights that are equiv to convolution:
%reshape SOH into matrix--should be identical to alternative W_conv assembly, and
%have same effect as convolution
W_conv_alt = reshape(W_conv_alt_SOH',x1_dim,x2_dim)'; 
W1 = W_conv_alt; %synonym

%compute network: can either use convnet w/ Kernel, or convnet_equiv w/ equiv W1
[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);
E_prev = E_new %previous error--must do better than this w/ update
E_new = err_fnc(Y_out,target_vals)
deltaE = E_new-E_prev
if (deltaE>0) %got worse!  back up and decrease eta
  display('got worse! backing up')
   bias2 = bias2_save;
   bias1_vec = bias1_save;
   W2 = W2_save;
   kernel_SOH = kernel_SOH_save;
  % E_new=E_prev;
   while (deltaE>0)
   	eta = 0.5*eta %shrink step size
   	dbias2 = -eta*deltavec2;
   	bias2 = bias2_save+dbias2;
   	
   	dbias1 = -eta*deltavec1;
                        bias1_vec = bias1_save+dbias1;	
	
   	dW2 = -  eta*W2sense; 
   	W2 = W2_save+dW2;
   	
   	dK_SOH=  - eta*dEdK_analytic;
  	kernel_SOH = kernel_SOH_save + dK_SOH; 
  	%convert new kernel into an equivalent W1 matrix 	
  	Kernel = reshape(kernel_SOH',Kcols,Krows)'; %new Kernel, expressed as a matrix_type
	K_overlay =  flipud(fliplr(Kernel));%overlay is rotated Kernel
	%equiv W1 from current values of Kernel/Overlay:
	k_overlay_SOH = reshape(K_overlay',1,Krows*Kcols);
	W_conv_alt_SOH = k_overlay_SOH*Kvecs_map; %compute W1 weights that are equiv to convolution:
	%reshape SOH into matrix--should be identical to alternative W_conv assembly, and
	%have same effect as convolution
	W1 = reshape(W_conv_alt_SOH',x1_dim,x2_dim)'; 
 	[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);
	E_new = err_fnc(Y_out,target_vals)
	deltaE = E_new-E_prev 	
   end
end
 %managed an improvement if made it to here; try increasing step size:
  eta = eta*2.0;
E_hist = [E_hist,E_new]; %keep a record of convergence
dE_expected
%if (deltaE<0) eta=2.0*eta;
%else
%	eta = eta/2.0
%end
display(' moving forward')
%kb = kbhit();  %can use this as a manual breakpoint
%compute sensitivities:

[deltasvec2, deltasvec1] = deltas_fnc(Y_out,target_vals, Gprime1,Gprime2,W2);
[W2sense,W1sense] = W_sensitivities_fnc(X0, X1, deltasvec1,deltasvec2);
[dedw_rows,dedw_cols] = size(W1sense);
W1_sensitivities_SOH = reshape(W1sense',1,dedw_rows*dedw_cols);
dEdK_analytic = %FIX ME!!!;
%size_dEdK = size(dEdK_analytic)

%sum of deltas over all images:
deltavec2 = deltasvec2*ones_cvec_nimages
deltavec1 = deltasvec1*ones_cvec_nimages;


%vary all params by some learning rate, eta:
%update layer2 bias(es):
dbias2 = -eta*deltavec2
%dbias2=0 %DEBUG OVERRIDE
bias2_save = bias2;
bias2 = bias2+dbias2
dE_expected_dbias2 = deltavec2*dbias2

dbias1 = -eta*deltavec1;
bias1_save = bias1_vec;
%dbias1 = 0*deltavec1; %DEBUG OVERRIDE
bias1_vec = bias1_vec+dbias1;
dE_expected_dbias1 = deltavec1'*dbias1

 %update wts
 dW2 = -  eta*W2sense;  %perturbations of output-layer weights
 %dW2 = 0*dW2; %DEBUG OVERRIDE
 W2_save = W2;
 W2 = W2 +dW2;  %here is the backprop update
 W2_SOH = W2; %W2 is already a row vector;  reshape(W2',1,npix_featuremap*num_image_pixels);
 W2sense_SOH =W2sense; % also already a row vector; reshape(W2sense',1,npix_featuremap*num_image_pixels);
 dW2_SOH = dW2; %already a row vector; reshape(dW2',1,npix_featuremap*num_image_pixels);
 dE_expected_dW2 = W2sense_SOH*dW2_SOH'
 
 %finally, perturbations of kernel:
 kernel_SOH = reshape(Kernel',1,Krows*Kcols);
 dK_SOH=  - eta*dEdK_analytic;
 kernel_SOH_save = kernel_SOH;
 %dK_SOH = dK_SOH*0; %DEBUG SUPPRESS
 kernel_SOH = kernel_SOH+ dK_SOH; %update the kernel
 Kernel = reshape(kernel_SOH',Kcols,Krows)'; %new Kernel, expressed as a matrix_type
 dE_expected_dK = dEdK_analytic*dK_SOH'
 
%expected influences: dE = deltavec2*dbias2 + deltavec1*dbias1 + W2sense_SOH*dW2_SOH
%   + dEdK_analytic * dK
display('expected influence next iter: ')
dE_expected = dE_expected_dbias2+dE_expected_dbias1+dE_expected_dW2+dE_expected_dK


end
plot(E_hist)

