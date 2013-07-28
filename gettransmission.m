function t = gettransmission(dark, A, alpha);
    
    if (~exist('alpha','var'))
        alpha=0.95;
    end  
    if (isempty(alpha))
        alpha=0.95;
    end
    
    [h,w,c]=size(dark);
    winI = reshape(dark, h * w, 1);
    t = 1 - alpha * winI / A;
%     winI = reshape(t, h, w);
%     figure, imshow(winI);
    
return;