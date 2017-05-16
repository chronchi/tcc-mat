function regular(x)
  (k,a) = size(x)
  b = zeros(k,a)
  for i = 1:a
    if x[:,i] == x[1,i]*ones(k)
      b[:,i] = x[:,i]
    else
      b[:,i] = (x[:,i] .- mean(x[:,i])) ./std(x[:,i])
    end
  end
  return b
end
