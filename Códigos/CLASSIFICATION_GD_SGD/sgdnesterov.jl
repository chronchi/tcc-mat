function sigmoid(x)
    #sigmoid function
    y = 1/(1+exp(-x))
    return y
end

predicte(w,x) = sigmoid(dot(w,x))

function gradsgdclass(w,x,y,i)
    dti = view(x,:,i)
    pdi = predicte(w, dti)
    a = (pdi - y[i]) * dti
    return a
end

function sgdnestclass(w_init,x,y,classes; num_of_epochs = 5,
  η = 1e-4, γ = .99, ν = .999, ϵ = 1e-8)

  (r,s) = size(x)
  w = copy(w_init)
  w_new = zeros(length(w))

  iter = 0
  #max_epoch = round(Int, ceil(max_iter/s))

  if classes == 2
    println("cant do it, dont even try")
  else
    classifiersvector = zeros(Float64, r, classes)
    for i = 1:classes
      multipliedμ = 1.0
      m = zeros(length(w))
      n = 0.0
      gw = ones(length(w))
      z = zeros(s)
      r = 0.0
      for j = 1:s
        if y[j] == i
          z[j] = 1.0
        else
          z[j] = 0.0
        end
      end
      for epoch = 1:num_of_epochs
        idx = collect(1:s)
        shuffle!(idx)
        for k = 1:s
          exp = convert(Float64, iter/250)
          μ = γ * (1.0 - .5 * .96 ^ exp)
          multipliedμ = multipliedμ * μ
          gw = gradsgdclass(w,x,z,idx[k])
          ghat = gw./(1-multipliedμ)
          m = γ * m - (1-γ) * gw
          mhat = m/(1-multipliedμ)
          n = ν * n + (1-ν) * norm(gw)^2
          nhat = n/(1-ν^iter)
          mstrich = (1-μ) * ghat + μ * mhat
          w = w - η * mstrich / (sqrt(nhat) + ϵ)
          iter += 1
        end
        classifiersvector[:,i] = w
        iter = 0
      end
    end
  end
  return classifiersvector
end
