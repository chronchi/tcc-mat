#REDE NEURAL ARTIFICIAL PARA A CLASSIFICAÇÃO DE IMAGENS


#X: matriz de dados, y: rótulos
#K: quantidade de classes
#size_hl: quantidade de neuronios em cada layer, incluindo input e output

function sigmoid(z)
  return 1./(1+exp(-z))
end

function sigmoidGradient(z)
  return sigmoid(z) .* (ones(size(z)) .- sigmoid(z))
end

function backpropnn(xtrain, ytrain, K , size_hl; alpha = .3, lambda = 1e-4,
 max_epoch = 1000)

  m,n = size(xtrain)

  W = rand((size_hl[2:end]'*size_hl[1:(end-1)])[1]);

  b = size_hl[2]*size_hl[1]
  c = b + size_hl[3]*size_hl[2]

  Theta1 = reshape(W[1:b], size_hl[2], size_hl[1]);
  Theta2 = reshape(W[(1+b):(b+ size_hl[3]*size_hl[2])], size_hl[3], size_hl[2]);
  Theta3 = reshape(W[(1+c):(c+ size_hl[4]*size_hl[3])], size_hl[4], size_hl[3]);

	b1 = rand(size(Theta1,1))
	b2 = rand(size(Theta2,1))
	b3 = rand(size(Theta3,1))

  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));
  Delta3 = zeros(size(Theta3));

  bdelta1 = zeros(length(b1))
  bdelta2 = zeros(length(b2))
  bdelta3 = zeros(length(b3))

  for epoch = 1:max_epoch
    for i = 1:m
      a_1 = xtrain[i,:]
      z_2 = Theta1 * a_1 + b1
      a_2 = sigmoid(z_2)
      z_3 = Theta2*a_2 + b2
      a_3 = sigmoid(z_3)
      z_4 = Theta3*a_3 + b3
      a_4 = sigmoid(z_4)
      y = zeros(length(a_4),1);
      y[ytrain[i]] = 1
      delta4 = (a_4 .- y);
      delta3 = (Theta3' * delta4) .* sigmoidGradient(z_3);
      delta2 = (Theta2' * delta3) .* sigmoidGradient(z_2);
      Delta1 += delta2 * a_1';
      Delta2 += delta3 * a_2';
      Delta3 += delta4 * a_3';
      Theta1 += - alpha * (1/m) * (Delta1 + lambda*Theta1);
      Theta2 += - alpha * (1/m) * (Delta2 + lambda*Theta2);
      Theta3 += - alpha * (1/m) * (Delta3 + lambda*Theta3);
	    bdelta1 += delta2
      bdelta2 += delta3
      bdelta3 += delta4
      b1 += -alpha * (1/m) * bdelta1
      b2 += -alpha * (1/m) * bdelta2
      b3 += -alpha * (1/m) * bdelta3
      if i%m == 0
        accuracy = 0
        for j = 1:m
           a = sigmoid(Theta3 * sigmoid(Theta2*sigmoid(Theta1*xtrain[j,:] + b1) + b2) + b3)
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
  return Theta1, Theta2, Theta3, b1, b2, b3
end

function prednn(x,y,K,size_hl;return_values = false,alpha = .3,
lambda = 1e-4, max_epoch = 1000)
  model = backpropnn(x,y,K,size_hl,alpha,lambda,max_epoch)
  multi = sigmoid(model[3]*sigmoid(model[2]*
  sigmoid(model[1]*x+model[4])+model[5])+model[6])
  results = zeros(size(multi))
  for i = 1:size(results,2)
    a = find(maximum(results[:,i]))[1]
    multi[a,i] = 1
  end
  accuracy = 0
  for i = 1:length(y)
    a = find(maximum(multi[:,i]))
    if y[i] == a
      accuracy += 1
    end
  end
  accuracy = accuracy/length(y)
  println("Test accuracy is" $accuracy)
  if return_values == true
    return multi
  end
end
