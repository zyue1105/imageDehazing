if (~exist('epsilon','var'))
  epsilon=[];
end
if (~exist('win_size','var'))
  win_size=[];
end

I = double(imread(img_name))/255;
[h, w, c] = size(I);
dark = getdarkchannel(I, 7);

A = getairlight(dark, I);
L = getLaplacian(I,1e-7,1);
t = gettransmission(dark, A, 0.95);
t = softmatting(L, t, 1e-4, 1e-8, 10000);
% winI = reshape(t, h, w);
% figure, imshow(winI);
res = recover(I, t, A, 0.1);
figure, imshow(res);

