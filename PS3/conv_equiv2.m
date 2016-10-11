%pgm to demo equivalence of conv2 and multiplication by Wc
%try to auto-gen Wc from K and A--stride implicitely=1
clear all
A = rand(4,4)
[NrowsA,McolsA] = size(A)
Ncomponents = NrowsA*McolsA
A_soh =reshape(A',[1,Ncomponents]) %"strung out horizontally"
%K = [1,-1, 0; 0,-1,1]
K=[1,2,3;4,5,6]
Kernel=K; %synonym

%sparse matrix attempt:
%specify the non-zero terms of kernel:
rvec = [1,1,2,2]
cvec = [1,2,2,3]
datavec = [1,-1,-1,1]
spK = sparse(rvec,cvec,datavec)


Kcols=[1,2];
[dummy,spncols] = size(Kcols);
Krows=[[1,2],[2,3]]
Kdata=[[1,-1],[-1,1]]
K2=zeros(2,3);

K_soh = reshape(K',[1,6]) %also strung out horizontally
B = conv2(A,K,'valid') %do 2-D convolution
figure(1)
imshow(B)
title('image map w/ conv2')
%equiv mapping: build a Wc matrix
%testing w/ K 2x3
Wc1 = [K(1,1), K(1,2), K(1,3), 0, K(2,1),K(2,2),K(2,3),0, 0,0,0,0, 0,0,0,0]
Wc2 = circshift(Wc1,[1,1]) %this one valid

Wc3 = circshift(Wc2,[1,3]) %shift 3 times to skip 2 invalid kernel shifts
Wc4 = circshift(Wc3,[1,1]) %

Wc5 = circshift(Wc4,[1,3]) %skip 2 invalid kernel shifts again
Wc6 = circshift(Wc5,[1,1]) %

Wc = [Wc1;Wc2;Wc3;Wc4;Wc5;Wc6]
[W_conv,x1_dim,x2_dim]= W_conv_equiv(A,K)
 [W_conv2,x1_dim,x2_dim,W_conv_alt,Kvecs_map]=W_conv_equiv2(A,K);
 
 %here is how to reconstitute a W_conv using the static Kvecs_map
 [Krows,Kcols] = size(Kernel)
 kvec_SOH = reshape(Kernel',1,Krows*Kcols);
 W_conv_alt_SOH = kvec_SOH*Kvecs_map;
%reshape SOH into matrix--should be identical to alternative W_conv assembly, and
%have same effect as convolution
W_conv_reconstituted = (reshape(W_conv_alt_SOH',x1_dim,x2_dim))'
%W_conv_reconstituted = W_conv_reconstituted' %transpose

W_conv2
W_conv_alt
W_err = Wc-W_conv_alt %this is all zeros

conv_equiv_vec = Wc*A_soh'
image_equiv= reshape(conv_equiv_vec,2,3)'
figure(2)
imshow(image_equiv)
title('image equiv')
diff = image_equiv - B %shows two ops are identical

