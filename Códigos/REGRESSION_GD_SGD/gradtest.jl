predict(x,w) = dot(x,w)

function lossfunction(w,x,y)
  m = size(x,1)
  z = zeros(m,1)
  b = ones(m,1)
  x = [b x]
  for i = 1:m
    z[i] = (predict(x[i,:],w) - y[i])^2
    b = (1/m) * sum(z[i])
    return b
  end
end

function gradient_of_loss(w,x,y,α = 0.01)
    m = size(x,1)
    k = size(x,2)
    GRAD = zeros(k,1)
    for j = 1:k
      for i = 1:m
        GRAD[j] = (1/m) * sum(predict(w,x[i,:])-y[i])*x[i,j]
      end
    end
    return GRAD
end

function trainer_grad(x,y,loss,grad, max_iter = 10000, tol=1e-4, α = 0.01)
    m = size(x,1)
    b = ones(m,1)
    x = [b x]
    k = size(x,2)
    w = rand(k,1)
    iter = 0
    lw = loss(w,x,y)
    gw = grad(w,x,y)
    while norm(gw) > tol && iter < max_iter

      w -= α * grad(w)
      lw = loss(w)
      gw = grad(w)
      iter += 1
    end

    return w, iter
end
