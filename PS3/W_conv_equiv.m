%fnc to get equivlent W matrix s.t. feature_vec = W*image_vec,
%given sample image and 2-D kernel
function [W_conv,x1_dim,x2_dim]= W_conv_equiv(sample_image,kernel)
[NROWS,NCOLS] = size(sample_image)
[Krows,Kcols] = size(kernel)
sample_featuremap = conv2(sample_image,kernel);
[FMrows,FMcols] = size(sample_featuremap)
%how many convolution locations are valid?
nvalid = (NCOLS-Kcols+1)*(NROWS-Krows+1)
x1_dim = NROWS*NCOLS
x2_dim = nvalid
%W_conv=zeros(x2_dim,x1_dim);

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
