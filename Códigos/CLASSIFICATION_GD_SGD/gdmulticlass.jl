#function gradclass(w,x,y)
#  m = length(y)
#  n = length(w)
#  a = zeros(n)
#  for j = 1:n
#    for i = 1:m
#      dti = view(x,:,i)
#      pdi = dot(w, dti)
#      a[j] += (pdi - y[i])*x[j,i]
#    end
#  end
#  return a
#end

function gradclass(w,x,y)
  return x'*x*w - x'*y
end

function gdclass(w_init,X,y,classes;max_iter=10000, α = 1.0, tol = 1e-4,max_time = 60.0,λ=1e-4)
    #starting the parameters
    m,n = size(X)
    w = copy(w_init)
    c = classes
    classifiersvector = zeros(n, c)
    max_epoch = round(Int, ceil(max_iter/m))

    #now we look through the entire dataset max_epoch times
    if classes > 1
    for i = 1:c
      z = zeros(m)
      for j = 1:m
        if y[j] == i
          z[j] = 1.0
        else
          z[j] = 0.0
        end
      end

      gw = ones(m)
      w_new = zeros(length(w))

      start_time = time()
      elapsed_time = 0.0

      #iter = 0

      for k = 1:max_epoch
      #while norm(gw) > tol  && iter < max_iter
          gw = gradclass(w,X,z)
          w_new = w - (α/m) * gw
          w = w_new
          #α = 0.7*α
          #iter += 1
          #elapsed_time = time() - start_time
          #if elapsed_time > max_time
          #  break
          #end
      end
      classifiersvector[:,i] = w
    end
    return classifiersvector
  else
    iter = 0
    gw = ones(length(w))
    f = zeros(Float64, max_epoch*m)
    for epoch = 1:max_epoch
        iter += 1
        gw = gradclass(w,X,y)
        w_new = w - α * gw
        w = w_new
        #α = 0.7*α
        #elapsed_time = time() - start_time
        #if elapsed_time > max_time
        #  break
        #end
        #if eval == true
        #  f[iter] = cost_function_classification(w,X,y)
        #end
    end
    #if eval == true
      #return w, f, iter
    #else
      return w
    end
end
