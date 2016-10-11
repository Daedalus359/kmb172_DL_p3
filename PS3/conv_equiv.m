%pgm to demo equivalence of conv2 and multiplication by Wc
A = rand(4,4)
A_sov = reshape(A,[16,1]) 
A_soh =reshape(A',[1,16]) %"strung out horizontally"
K = [1,-1; -1 1]
K_soh = reshape(K',[1,4]) %also strung out horizontally
B = conv2(A,K,'valid') %do 2-D convolution
figure(1)
imshow(B)
%equiv mapping: build a Wc matrix

Wc1 = [K(1,1), K(1,2), 0, 0, K(2,1),K(2,2),0,0, 0,0,0,0, 0,0,0,0]
Wc2 = circshift(Wc1,[1,1])
Wc3 = circshift(Wc2,[1,1])

Wc4 = circshift(Wc3,[1,2])
Wc5 = circshift(Wc4,[1,1])
Wc6 = circshift(Wc5,[1,1])

Wc7 = circshift(Wc6,[1,2])
Wc8 = circshift(Wc7,[1,1])
Wc9 = circshift(Wc8,[1,1])

Wc = [Wc1;Wc2;Wc3;Wc4;Wc5;Wc6;Wc7;Wc8;Wc9]
conv_equiv_vec = Wc*A_soh'
image_equiv= reshape(conv_equiv_vec,3,3)'
figure(2)
imshow(image_equiv)
diff = image_equiv - B %shows two ops are identical

