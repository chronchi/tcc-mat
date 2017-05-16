function lossfunction(w,x,y)
  m = size(x,1)
  z = zeros(m,1)
  b = ones(m,1)
  x = [b x]
  for i = 1:m
    z[i] = (dot(x[i,:],w) - y[i])^2
    b = (1/m) * sum(z[i])
    return b
  end
end

function gradsgdclass(w,x,y,i)
    dti = view(x,:,i)
    pdi = dot(w, dti)[1]
    a = (pdi - y[i]) * dti
    return a
end

function trainer_grad(x,y,loss,grad, epoch = 100, tol=1e-4, α = 0.01)
    m = size(x,1) 
    x = [ones(m,1) x]
    w = rand(size(x,2),1)
    iter = 0
    lw = loss(w,x,y)
    for j = 1:epoch
    idx = collect(1:m)
    shuffle!(idx)
    for i = 1:m
      w -= α * grad(w,x,y,i)
      gw = grad(w)
      iter += 1
      if norm(gw) < tol
        break
      end 
    end
    end
    return w, iter
end
