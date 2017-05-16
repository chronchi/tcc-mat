#array splitter

function splitter(X,num_parts::Int,block::Int)
  if block > num_parts
    return println("error, the block is greater than the folds")
  end
  m,n = size(X)
  roundup = convert(Int, (m/num_parts))
  q = block*roundup
  u = q + roundup

  if block == 1
    xtrain = X[(roundup+1):end,:]
    xtest = X[1:roundup,:]
  elseif block == num_parts
    xtrain = X[1:(m-roundup),:]
    xtest = X[(m-roundup+1):end,:]
  else
    xtrain = zeros(m-roundup,n)
    xtrain[1:q-roundup,:] = X[1:q-roundup,:]
    xtrain[q+1-roundup:end,:] = X[q+1:end,:]
    xtest = X[(q+1-roundup:q),:]
  end
  return xtrain, xtest
end
