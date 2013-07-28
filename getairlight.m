function A = getairlight(dark, I)
    [h,w] = size(dark);
    tmp = reshape(dark', h * w, 1);
    [tmp, ind] = sort(tmp, 'descend');
    A = 0;
    for i = 1:max(1, h * w * 0.001) 
        row = floor((ind(i) - 1) / w) + 1;
        col = mod((ind(i) - 1), w) + 1;
        A = max(A, (I(row, col, 1) + I(row, col, 2) + I(row, col, 3)) / 3);
    end;
return;