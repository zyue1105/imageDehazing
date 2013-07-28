function dark=getdarkchannel(I, N)
    if (~exist('N','var'))
        N = 7;
    end  
    if (isempty(N))
        N = 7;
    end
    mI = min(I, [], 3);
    [h,w,c] = size(I);
    dark = zeros(h, w);
    
    for i = 1:h
        for j = 1:w
            winI = mI(max(1,i-N):min(i+N,h),max(1,j-N):min(w,j+N),:);%È¡ÁìÓòÏñËØ
            dark(i, j) = min(min(winI));
        end;
    end;
    %figure, imshow(dark);
    
return;