%create training images that are noisy, but include a distinguishing feature;
%I will choose to make a horizontal, bright line segment somewhere in the scene
%convolution should find this feature
NPIX = 28
NIMAGES = 100
  images_w_feature = zeros(NIMAGES,NPIX,NPIX);
  images_wo_feature = zeros(NIMAGES,NPIX,NPIX);
for i_image=1:NIMAGES
  figure(1)
  im_noisy = rand(NPIX,NPIX);
  imshow(im_noisy)
  im_feature = add_lineseg(im_noisy);
  figure(2)
  imshow(im_feature)
  sleep(0.1)
  images_w_feature(i_image,:,:)=im_feature;
  images_wo_feature(i_image,:,:)=im_noisy;
end
%fname="cnn_data"
save cnn_data images_w_feature images_wo_feature

