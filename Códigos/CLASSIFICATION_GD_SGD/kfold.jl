#function for k-fold-cross validation
include("gdcmulticlass.jl")
include("sgdmulticlass.jl")
include("gdmulticlass.jl")
include("split.jl")
include("runpred.jl")
gi
using MLBase

function kfold(w_init::Vector,xdata::Matrix,ydata::Array,
  classes::Int,num_folds::Int, sgdorgd::Function, symbols::Array;
  max_iter::Int = 100000,α::Float64 = .3, eval = false, lambda = 1e-4)

  # first we split our dataset into k parts
  #xdata = [ones(size(xdata,1)) xdata]
  m = size(xdata,1)
  n = length(w_init)

  append = [xdata ydata]
  random_vector = sample(1:m,m,replace = false)
  append = append[random_vector,:]
  meanerror = 0.0
  meancorr = 0.0
  parameters = zeros(n,num_folds)


  for i = 1:num_folds
    train, test = splitter(append,num_folds,i)
    xtrain = train[:,1:n]
    ytrain = convert(Array{Int64},train[:,n+1])
    xtest = test[:,1:n]
    ytest = test[:,n+1]
    ytest = convert(Array{Int64}, ytest)
    a = sgdorgd(w_init, xtrain, ytrain, classes, α = α, max_iter = max_iter,λ=lambda)
    if classes > 1
      results = predicting(xtest,a,symbols)
      results = convert(Array{Int64}, results)
      error = errorrate(ytest,results)
      corr = correctrate(ytest,results)
      println("The error for the $i-fold is $error, accuracy is given by $corr")
      #results = zeros(size(predictor(xtest,a),1))
    else
      b = zeros(size(xtest,1))
      for i = 1:size(xtest,1)
        b[i] = dot(a,xtest[i,:])
      end
      results = round(sigmoid.(b))
      results = convert(Array{Int64}, results)
      error = errorrate(ytest,results)
      println("The error for this fold is $error")
      results = zeros(length(results))
    end
    meanerror += error
    meancorr += corr
    #parameters[:,i] = a
  end
  mean_error = meanerror/num_folds
  mean_corr = meancorr/num_folds
  println("The average error is $mean_error and accuracy is given by $mean_corr")
  #return parameters
end
