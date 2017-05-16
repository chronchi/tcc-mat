using RDatasets

iris = dataset("datasets", "iris")

xtrain = convert(Array{Float64}, iris[:,1:4])

function convertOneHot(y::Any, number_of_classes::Int64;vector=true)
  if vector == true
    a = length(y)
    S = []
    for i = 1:a
      if !(y[i] in S)
        S = [S;y[i]]
      end
    end
    b = length(S)
    d = zeros(Int,a)
    for j = 1:b
      c = find(y .== S[j])
      d[c] = j
    end
    return d
  else
    a = length(y)
    S = []
    for i = 1:a
      if !(y[i] in S)
        S = [S;y[i]]
      end
    end
    b = number_of_classes
    d = zeros(Int,a,b)
    for j = 1:length(S)
      c = find(y .== S[j])
      d[c,j] = 1
    end
    return d
  end
end

ytrain = convertOneHot(iris[:,5],3)

random_vector = sample(1:150,10,replace=false)
xtest = xtrain[random_vector,:]
ytest = ytrain[random_vector]

xtrain = xtrain[setdiff(collect(1:150),random_vector),:]
ytrain = ytrain[setdiff(collect(1:150),random_vector)]
