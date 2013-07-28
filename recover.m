function res = recover(I, tran, A, tx)
    if (~exist('tx','var'))
        tx=0.1;
    end  
    if (isempty(tx))
        tx=0.1;
    end
    
    [h,w,c] = size(I);
    res = zeros(h, w, c);
    tran = reshape(tran, h, w);
    for(i = 1:h)
        for(j = 1:w)
            for(k = 1:3)
                tp = (I(i,j,k) - A) / max(tx, tran(i,j)) + A;
                if(tp > 1.0)
                    tp = 1.0;
                end;
                res(i,j,k) = tp;
            end;
        end;
    end;

return;