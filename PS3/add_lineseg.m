function [out_image] = add_lineseg(in_image)
[npix,npix] = size(in_image);
width = 11; %arbitrary choice for lineseg length
lineseg = ones(1,width);
irow = round(rand(1)*(npix-3))+2;
%want rand number from 6 to npix-6 for center of horiz line
jcol_ctr = round(rand(1)*(npix-12))+6;

%install the line:
out_image = in_image;
out_image(irow,jcol_ctr-5:jcol_ctr+5)= lineseg;