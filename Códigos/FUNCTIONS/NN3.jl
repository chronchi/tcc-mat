function backpropnn3(xtrain, ytrain, K , size_hl; alpha = .3, lambda = 1e-4,
 max_epoch = 1000)

  m,n = size(xtrain)

  W = rand((size_hl[2:end]'*size_hl[1:(end-1)])[1]);

  b = size_hl[2]*size_hl[1]
  c = b + size_hl[3]*size_hl[2]

  Theta1 = reshape(W[1:b], size_hl[2], size_hl[1]);
  Theta2 = reshape(W[(1+b):(b+ size_hl[3]*size_hl[2])], size_hl[3], size_hl[2]);

  b1 = rand(size(Theta1,1))
	b2 = rand(size(Theta2,1))

  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));

  bdelta1 = zeros(length(b1))
  bdelta2 = zeros(length(b2))

  for epoch = 1:max_epoch
    for i = 1:m
      a_1 = xtrain[i,:]
      z_2 = Theta1 * a_1 + b1
      a_2 = sigmoid(z_2)
      z_3 = Theta2*a_2 + b2
      a_3 = sigmoid(z_3)
      y = zeros(length(a_3),1);
      y[ytrain[i]] = 1
      delta3 = (a_3 .- y);
      delta2 = (Theta2' * delta3) .* sigmoidGradient(z_2);
      Delta1 += delta2 * a_1';
      Delta2 += delta3 * a_2';
      Theta1 += - alpha * ((1/m)*Delta1 + lambda*Theta1);
      Theta2 += - alpha * ((1/m)*Delta2 + lambda*Theta2);
      bdelta1 += delta2
      bdelta2 += delta3
      b1 += -alpha * (1/m) * bdelta1
      b2 += -alpha * (1/m) * bdelta2
      if i%m == 0
        accuracy = 0
        for j = 1:m
           a = sigmoid(Theta2*sigmoid(Theta1*xtrain[j,:] + b1) + b2)
           #a = convert(Array{Int64}, a)
           finder = find(a.==maximum(a))[1]
           if finder == ytrain[j]
            accuracy += 1
           #else
           # println("$a")
           end
        end
        accuracy = accuracy/m
        println("$accuracy")
      end
    end
  end
  return Theta1, Theta2, b1, b2
end
