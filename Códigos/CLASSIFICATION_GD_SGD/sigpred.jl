function sigmoid(x)
    #sigmoid function
    y = 1/(1+exp(-x))
    return y
end

predictor(w,x) = sigmoid(dot(w,x))
