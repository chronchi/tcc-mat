function cost_function_classification(w,x,y)
    m = length(y)
    a = 0.0
    for i = 1:m
        dti = view(x, :, i)
        pdi = sigmoid.(w'*dti)
        a += y[i]*log(pdi) + (1-y[i])*log(1-pdi)
    end
    return (-a/m)[1]
end
