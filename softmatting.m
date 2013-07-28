function tran = softmatting(L, t, lamda, tol, maxit);
    if (~exist('lamda','var'))
        lamda = 1e-4;
    end  
    if (isempty(lamda))
        lamda = 1e-4;
    end
    if (~exist('tol','var'))
        tol = 1e-6;
    end  
    if (isempty(tol))
        tol = 1e-6;
    end
    if (~exist('maxit','var'))
        maxit = 10000;
    end  
    if (isempty(maxit))
        maxit = 10000;
    end
    [n, m] = size(L);
    b = t * lamda;
    A = speye(n) * lamda;
    L = L + A;
    tran = pcg(L, b, tol, maxit);
return;