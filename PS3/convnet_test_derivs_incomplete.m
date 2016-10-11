%convnet_main:
clear all
load cnn_data


[N_IMAGES,NROWS,NCOLS] = size(images_w_feature)
targets_w_feature = ones(N_IMAGES,1); %col vec of corresponding desired y_out values for images w/ feature
[N_IMAGES_WO_FEATURES,NROWS,NCOLS] = size(images_w_feature)
targets_wo_feature = zeros(N_IMAGES_WO_FEATURES,1); %col vec of corresponding desired y_out values for images w/o feature

num_image_pixels = NROWS*NCOLS

krows=2 %3
kcols= 3% 11
K_overlay = rand(krows,kcols)
%note difference: K is an overlay mask, but corresponding conv kernel is flipped right/left AND upside down
Kernel= flipud(fliplr(K_overlay)) %kernel is related to overlay

%do a conv2, just to get dimensions:
 test_image = squeeze(images_w_feature(1,:,:)); 
 test_image_SOH= reshape(test_image',1,num_image_pixels);
featuremap = conv2(test_image,Kernel,'valid');
[fm_rows,fm_cols] = size(featuremap);
npix_featuremap = fm_rows*fm_cols;
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
 [NPIX,N_ALL_IMAGES] =   size(X0)
 %images are now all column vecs; create corresponding targets in corresponding columns:
 target_vals = ones(1,N_IMAGES);
 target_vals = [target_vals,zeros(1,N_IMAGES_WO_FEATURES)];
 
 %that's all the preliminaries; simulate and train networks
%try on a single image:
[y_out1,x1,gprime1_vec,gprime2] = convnet(test_image,Kernel,bias1_vec,W2,bias2);
y_out1
%SOH means "strung out horizontally"
image_vec = test_image_SOH';
[y_out2,X1,Gprime1,Gprime2] = convnet_equiv(image_vec,W1,bias1_vec,W2,bias2);
y_out2
dy = y_out2-y_out1

%do all the images at once, in a batch:
[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);

%compute error(s):
E = err_fnc(Y_out,target_vals)
%[deltavec2, deltavec1] = deltas_fnc(Y_out,target_vals, Gprime1,Gprime2,W2);
%[W2sense,W1sense] = W_sensitivities_fnc(X0, X1, deltavec1,deltavec2);

%compute derivatives:
[deltavec2, deltavec1] = deltas_fnc(Y_out,target_vals, Gprime1,Gprime2,W2);
deltavec2_size = size(deltavec2)
deltavec1_size = size(deltavec1)
[W2sense,W1sense] = W_sensitivities_fnc(X0, X1, deltavec1,deltavec2);
W2sense_size = size(W2sense)
W1sense_size = size(W1sense)

%test derivatives
pct_errs=[];
for k=1:20
i=1;
j=randi(W2sense_size(2))
dW2_analytic = W2sense(j)
dW2_numerical = compute_numerical_deriv_W2(j,X0,W1,bias1_vec,W2,bias2,target_vals)
dW_pct_diff = (dW2_analytic-dW2_numerical)/dW2_numerical
pct_errs=[pct_errs,dW_pct_diff];
end
pct_errs
display('check above W2 sensitivities; hit a key to move on')
kb = kbhit();
%change network wts
bias1_vec = 2*rand(npix_featuremap,1)-ones(npix_featuremap,1);
W2 = (2*rand(1,npix_featuremap)-ones(1,npix_featuremap))*0.1;
bias2 = 2*rand(1,1)-1;
W1 =( 2*rand(W1rows,W1cols )-ones(W1rows,W1cols ))*0.07;
%compute the sensitivities
[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(X0,W1,bias1_vec,W2,bias2);
[deltavec2, deltavec1] = deltas_fnc(Y_out,target_vals, Gprime1,Gprime2,W2);
[W2sense,W1sense] = W_sensitivities_fnc(X0, X1, deltavec1,deltavec2);
pct_errs_W1=[];
for k=1:20  %select kernel components at random to eval perturbations
ival= randi(W1sense_size(1))
jval= randi(W1sense_size(2))
dW1_analytic = W1sense(i,j)
%slow, approximate numerical estimate of deriviative with respect to chosen, single W1 component:
dW1_numerical = compute_numerical_deriv_W1(i,j,X0,W1,bias1_vec,W2,bias2,target_vals)
dW_pct_diff = (dW1_analytic-dW1_numerical)/dW1_numerical
pct_errs_W1=[pct_errs_W1,dW_pct_diff];
end
pct_errs_W1
display('check above W1 sensitivities; hit a key to move on')
kb = kbhit();

%%  The above should demonstrate that the analytic derivatives are very close to the approximate 
%numerical derivatives for perturbations of individual W components

%now, test derivs w/rt kernel elements:
epsk = 0.0001 %magnitude of perturbation of kernel component to evaluate
while true %loop through random spot checks
display('ready to spot check kernel sensitivities; hit a key')
kb = kbhit();
%"overlay" is same dimensions as kernel, but rotated 180deg; overlay is more intuitive than kernel
K_overlay = rand(krows,kcols)
[Krows,Kcols] = size(K_overlay);
%change network wts
bias1_vec = 2*rand(npix_featuremap,1)-ones(npix_featuremap,1);
W2 = 2*rand(1,npix_featuremap)-ones(1,npix_featuremap);
bias2 = 2*rand(1,1)-1;
%equiv W1: fully-connected weights that correspond to equivalent 2-D convolution operation
kvec_SOH = reshape(K_overlay',1,Krows*Kcols);
W_conv_alt_SOH = kvec_SOH*Kvecs_map;
%reshape SOH into matrix--should be identical to alternative W1 matrix, and
%have same effect as convolution
W1 = reshape(W_conv_alt_SOH',x1_dim,x2_dim)'; 

%note difference: K is an overlay mask, but corresponding conv kernel is flipped right/left AND upside down
Kernel= flipud(fliplr(K_overlay)) %kernel is related to overlay


%make a vector of same dimension as Kernel, strung out horizontally
%choose a single component, at random, of this vector to set to unity; all other terms = 0
select_dk_vec = zeros(1,krows*kcols);
krand = randi(krows*kcols)
select_dk_vec(krand) = 1; 
%compute the equivalent "overlay" matrix with a single non-zero term:
dk_overlay = reshape(select_dk_vec',kcols,krows)'
%compute the equivalent Kernal matrix for this single-component selection;
%this is the term that will be perturbed
dKernel_map = flipud(fliplr(dk_overlay))

i_image = randi(N_IMAGES) %select an image at random from the set
test_image = squeeze(images_w_feature(i_image,:,:));  %convert to vector
target_val = target_vals(i_image) %and get its corresponding target value (0/1)

%compute the output error for two different networks: the original network, and
%a network with only one component of the Kernel perturbed.  Infer the error derivative with respect to this element
dEdK_numerical = compute_numerical_deriv_kernel(test_image,Kernel,dKernel_map,bias1_vec,W2,bias2,target_val)

x0=reshape(test_image',1,num_image_pixels)';

%x0 = X0(:,i_image); %same image, but expressed as a column vec
%compute sensitivities:
[Y_out,X1,Gprime1,Gprime2] = convnet_equiv(x0,W1,bias1_vec,W2,bias2);
[deltavec2, deltavec1] = deltas_fnc(Y_out,target_val, Gprime1,Gprime2,W2);
[W2sense,W1sense] = W_sensitivities_fnc(x0, X1, deltavec1,deltavec2);
[dedw_rows,dedw_cols] = size(W1sense);
W1_sensitivities_SOH = reshape(W1sense',1,dedw_rows*dedw_cols);
display('ideally, the next two values are close to each other, to confirm analytic vs numerical derivs')
%compute dE/dKij analytically:
%%%%  FIX ME!!!!!
dEdK_analytic = %USE W1_sensitivities and Kvecs_map to compute this
%compare this to the numerical estimate: the two should be close to each other
dEdK_numerical

end

