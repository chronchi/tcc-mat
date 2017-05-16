using HDF5, JLD
using Distributions
using TensorFlow
using DataFrames

#xtrain, ytrain = load("fold1xwb.jld")["xtrain"],load("fold1ywb.jld")["ytrain"];


path = "/home/carlos/Desktop/TCC/HASYv2/classification-task/fold-1"
data = readtable("train.csv")

symbol_id = readtable("symbols.csv")[:symbol_id]

xtrain = read_data(1024,path,data[:,1])
ytrain = changesymb(data[:,2],symbol_id)
ytrain = convertOneHot(ytrain,369,vector=false)

#Função que nos dá um batch por iteração

m,n = size(xtrain)

size_of_batch = 500
numb_of_batch = convert(Int64,round(m/size_of_batch)) - 1

function next_batch(xtrain, ytrain, batch_size, numb_of_batch, i)
  n = size(xtrain,1)
  if i < numb_of_batch
    vector = collect((i*batch_size):(i*batch_size+batch_size))
    return xtrain[vector,:], ytrain[vector,:]
  elseif i == numb_of_batch
    b = (numb_of_batch-1)*batch_size + batch_size + 1
    vector = collect(b:n)
    return xtrain[vector,:], ytrain[vector,:]
  end
end

session = Session(Graph())

function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end

function bias_variable(shape)
    initial = fill(Float32(.1), shape...)
    return Variable(initial)
end

function conv2d(x, W)
    nn.conv2d(x, W, [1, 1, 1, 1], "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end

x = placeholder(Float32)
y_ = placeholder(Float32)

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

x_image = reshape(x, [-1, 32, 32, 1])

h_conv1 = nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8*8*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = reshape(h_pool2, [-1, 8*8*64])
h_fc1 = nn.relu(h_pool2_flat * W_fc1 + b_fc1)

keep_prob = placeholder(Float32)
h_fc1_drop = nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 369])
b_fc2 = bias_variable([369])

y_conv = nn.softmax(h_fc1_drop * W_fc2 + b_fc2)

cross_entropy = reduce_mean(-reduce_sum(y_.*log(y_conv+1e-10), axis=[2]))

train_step = train.minimize(train.AdamOptimizer(.0005), cross_entropy)

correct_prediction = indmax(y_conv, 2) .== indmax(y_, 2)

accuracy = reduce_mean(cast(correct_prediction, Float32))

run(session, global_variables_initializer())

for j = 1:10
  for i in 1:numb_of_batch
    batch = next_batch(xtrain,ytrain,size_of_batch,numb_of_batch,i)
    if i%50 == 0
      train_accuracy = run(session, accuracy, Dict(x=>batch[1], y_=>batch[2], keep_prob=>1.0))
      info("step $(150*j + i), training accuracy $train_accuracy")
    end
    run(session, train_step, Dict(x=>batch[1], y_=>batch[2], keep_prob=>.5))
  end
end

datatest = readtable("test.csv")
xtest = read_data(1024,path,datatest[:,1])
ytest = changesymb(datatest[:,2],symbol_id)
ytest = convertOneHot(ytest,369,vector=false)

acuracia = 0

for i = 1:32
  batch = next_batch(xtest,ytest,size_of_batch,numb_of_batch,i)
  test_accuracy = run(session, accuracy, Dict(x=>batch[1], y_=>batch[2], keep_prob=>1.0))
  info("test accuracy $test_accuracy")
  acuracia += test_accuracy
end
acuracia = acuracia/32
info("Accuracy $acuracia")
