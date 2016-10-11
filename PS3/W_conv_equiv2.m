%fnc to get equivalent W matrix s.t. feature_vec = W*image_vec,
%given sample image and 2-D kernel
%in this version, 
%note: SOV may be faster than SOH; e.g. if A is NxM, a = reshape(A,NxM,1) is col vec, SOV,
% reshape(a,N,M) reinstates

%Kvecs_map injects kernel values into W_conv with: 
%W_conv_alt_SOH = kvec_SOH*Kvecs_map;
%reshape SOH into matrix--should be identical to alternative W_conv assembly, and
%have same effect as convolution
%W_conv_alt = reshape(W_conv_alt_SOH',x1_dim,x2_dim); 
%W_conv_alt = W_conv_alt' %transpose
%RETAIN Kvecs_map as static, so can reconstitute a W_conv quickly given new kernel vals
%ALSO use Kvecs_map to compute net effect of each convolution kernel from dE/dW_conv

function [W_conv,x1_dim,x2_dim,W_conv_alt,Kvecs_map]= W_conv_equiv2(sample_image,kernel)
[NROWS,NCOLS] = size(sample_image)
[Krows,Kcols] = size(kernel)
kernelvec_dim = Krows*Kcols
kvec_SOH = reshape(kernel',1,Krows*Kcols);
sample_featuremap = conv2(sample_image,kernel);
[FMrows,FMcols] = size(sample_featuremap)
%how many convolution locations are valid?
nvalid = (NCOLS-Kcols+1)*(NROWS-Krows+1)
x1_dim = NROWS*NCOLS
x2_dim = nvalid
%W_conv=zeros(x2_dim,x1_dim);

%kvec_map_row_i is a rowvec that inserts k_i into W_conv_SOH
%start w/ kernel(1):
kvec_map_empty = zeros(1,x1_dim);
kvec_map_pieces1 = kvec_map_empty;
kvec_map_pieces1(1) = 1;

Kvecs_map=[];
kernels_mat=zeros(nvalid,Krows,Kcols);

kkernel=1
for ik=1:Krows
  for jk=1:Kcols
     kernelij = zeros(Krows,Kcols);
     kernelij(ik,jk) = 1;
     kernels_mat(kkernel,:,:) = kernelij;
     kkernel=kkernel+1;
     end
    end
%kernels_mat
     
     
kmap_mat=[];
%kvec_map_pieces1 = kvec_map_empty;
%kvec_map_pieces1(k) = 1;	
for kkernel=1:kernelvec_dim

   %kkernel=6 %test
      W_row_pieces = zeros(Krows,NROWS);
      kernel_k = squeeze(kernels_mat(kkernel,:,:));
      W_row_pieces(1:Krows,1:Kcols) = kernel_k; %B = squeeze(A(:,8,:));  First kernel only
      W_conv_row = zeros(1,x1_dim);
for k=0:Krows-1
  W_conv_row(1,k*NCOLS+1:(k+1)*NCOLS)=W_row_pieces(k+1,:);	
end
 % W_conv_row

kmap_mat=[];
for irow=1:NROWS-Krows+1
   for nshifts=1:NCOLS-Kcols+1
      kmap_mat = [kmap_mat;W_conv_row];
      W_conv_row=circshift(W_conv_row,[1,1]);  %this takes care of sweep left-to-right	
   end
    W_conv_row=circshift(W_conv_row,[1,Kcols-1]);	
 end	
%kmap_mat
[kmap_rows,kmap_cols] = size(kmap_mat);
kvec_map1 = reshape(kmap_mat',1,kmap_rows*kmap_cols);
Kvecs_map = [Kvecs_map;kvec_map1];
end
Kvecs_map_size = size(Kvecs_map)
kvec_SOH_size = size(kvec_SOH)
%assemble W_conv_SOH from mapping vectors for each component of kernel
%here's the big deal: adds contributions from each kernel component
%USE THIS as well for combining sensitivities of W_conv coupling effects from individual kernel values
W_conv_alt_SOH = kvec_SOH*Kvecs_map;
%reshape SOH into matrix--should be identical to alternative W_conv assembly, and
%have same effect as convolution
W_conv_alt = reshape(W_conv_alt_SOH',x1_dim,x2_dim); 
W_conv_alt = W_conv_alt';


%build W_conv NOT using individual kvec injection maps:
W_row_pieces = zeros(Krows,NROWS);
W_row_pieces(1:Krows,1:Kcols) = kernel;

W_conv_row = zeros(1,x1_dim);
for k=0:Krows-1
  W_conv_row(1,k*NCOLS+1:(k+1)*NCOLS)=W_row_pieces(k+1,:);	
end
%example for image 4x4 and kernel 2x3
%Wc2 = circshift(Wc1,[1,1]) %this one valid
%Wc3 = circshift(Wc2,[1,3]) %shift 3 times to skip 2 invalid kernel shifts
%Wc4 = circshift(Wc3,[1,1]) %
%Wc5 = circshift(Wc4,[1,3]) %skip 2 invalid kernel shifts again
%Wc6 = circshift(Wc5,[1,1]) %
%shifts: start w/ Wc1; addl NCOLS-Kcols shifts to do;
%then finish shifting Nshifts = Kcols
%repeat this Ntimes = NROWS-Krows times
W_conv=[];
for irow=1:NROWS-Krows+1
   for nshifts=1:NCOLS-Kcols+1
      W_conv = [W_conv;W_conv_row];
      W_conv_row=circshift(W_conv_row,[1,1]);  %this takes care of sweep left-to-right	
   end
    W_conv_row=circshift(W_conv_row,[1,Kcols-1]);	
end	
