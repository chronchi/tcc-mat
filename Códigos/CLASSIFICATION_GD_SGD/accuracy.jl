function accuracysqm(predicted,test)
  a = 0.0
  for i = 1:length(test)
    a += (predicted[i] - test[i])^2
  end
  a = a/length(test)
  println("Your accuracy is $a%.")
end
