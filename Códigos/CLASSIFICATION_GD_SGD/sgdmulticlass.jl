function gradsgdclass(w,x,y,i)
    dti = view(x,:,i)
    pdi = dot(w, dti)
    a = (1/(1+e^(-pdi) - y[i])) * dti
    return a
end

function sgdclass(w_init,x,y,classes; max_iter = 10000, α = .01,
  eval = false, max_time = 10.0, λ = 1e-4)

  (m,n) = size(x)
  w = copy(w_init)
  w_new = zeros(length(w))

  iter = 0
  max_epoch = round(Int, ceil(max_iter/m))

  classesvector = collect(1:classes)

  if classes == 2
    println("cant do it, dont even try")
  else
    classifiersvector = zeros(Float64, n, classes)
    idx = collect(1:m)
    shuffle!(idx)
    for i = 1:classes
      v = zeros(n)
      z = zeros(m)
      r = 0.0
      for j = 1:m
        if y[j] == i
          z[j] = 1.0
        else
          z[j] = 0.0
        end
      end
      for epoch = 1:max_epoch
        for k = 1:m
          iter += 1
          #γ = α/(1 + λ * α * iter)
          gw = gradsgdclass(w,x',z,idx[k])
          #α = α/(1+α*iter)
          w = w - α*gw
          #w_new = w - γ * ((gw) + (λ/m) * w)
          #r_new = r + γ
          #v_new = r/r_new * v + (r_new - r)/r_new * w
          #w = w_new
          #r = r_new
          #v = v_new
        end
      end
      iter = 0
      classifiersvector[:,i] = w
    end
  end
  return classifiersvector
end
