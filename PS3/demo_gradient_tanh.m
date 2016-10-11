%demo gradient of tanh
%shows that for sigmoid = 0.5*(tanh(u)+1), 
% g' = 2*sigma*(1-sigma)

eps=0.000001
xvec=[]
errs=[]
for x=-1:0.01:1
	xvec=[xvec,x];
  sigma = 0.5*(tanh(x)+1)
  sig_eps = 0.5*(tanh(x+eps)+1);
  deriv_num = (sig_eps-sigma)/eps
  deriv_formula = 2*sigma*(1-sigma)
  err = deriv_num-deriv_formula
  errs=[errs,err];
  end
  figure(1)
  plot(xvec,errs)
  