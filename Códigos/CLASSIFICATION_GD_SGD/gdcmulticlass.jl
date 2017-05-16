function gradconj(w,x,y)
    return x'*(sigmoid.(x*w) .- y)
end

function gdcclass(w_init,X,y,classes;max_iter=10000, α = 1.0, tol = 1e-4,max_time = 60.0,λ=1e-4)
    #starting the parameters

  (m,n) = size(X)
  w = copy(w_init)
  iter = 0
  max_epoch = round(Int, ceil(max_iter/m))
  M = X'*X
  w_old = zeros(length(w))
  w_new = zeros(length(w))
  d = zeros(length(w))
  counter = 0
  if classes == 2
    println("cant do it, dont even try")
  else
    classifiersvector = zeros(Float64, n, classes)
    for i = 1:classes
      z = zeros(Int64,m)
      for j = 1:m
        if y[j] == i
          z[j] = 1
        else
          z[j] = 0
        end
      end
      d = gradconj(w,X,z)
      w_new = w - α*d
      w_old = w
      w = w_new
      while norm(gradconj(w,X,z)) > 1e-3
          β = (norm(gradconj(w,X,z))/norm(gradconj(w_old,X,z)))^2
          d = -gradconj(w,X,z) + β*d
          w_new = w + α*d
          w_old = w
          w = w_new
          counter += 1
          if counter > max_iter
            break
          end
      end
      classifiersvector[:,i] = w
      iter = 0
      w = copy(w_init)
    end
  end
  return classifiersvector
end
