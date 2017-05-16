function testbackprop(xtest,ytest,model)
  accuracy = 0
  m = length(ytest)
  for j = 1:m
    a = sigmoid(model[2]*sigmoid(model[1]*xtest[j,:] + model[3]) + model[4])
    finder = find(a.==maximum(a))[1]
    if finder == ytest[j]
      accuracy += 1
    end
  end
  accuracy = accuracy/m
  println("Your accuracy is of $accuracy on the training test")
end
