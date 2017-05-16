function predicting(x,a,y)
  m = size(x,1)
  results = zeros(m);
  n = size(a,2)
  expected = zeros(Float64, m,n);
  for j = 1:m
    for k = 1:n
      expected[j,k] = sigmoid(dot(a[:,k],x[j,:]))
    end
  end
  for i = 1:m
    for j = 1:n
      if maximum(expected[i,:]) == expected[i,j]
        results[i] = y[j]
      end
    end
  end
  return results
end
