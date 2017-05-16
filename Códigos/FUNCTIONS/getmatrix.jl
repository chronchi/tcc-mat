function getmatrix(listao, A)
  I,J = listao[:,1], listao[:,2]
  a = extrema(I)
  b = extrema(J)
  c = a[2] - a[1] + 1
  d = b[2] - b[1] + 1
  E = zeros(c,d)
  mi = length(I)
  for k in 1:mi
    E[I[k]-a[1]+1,J[k]-b[1]+1] = A[I[k],J[k]]
  end
  return E
end
