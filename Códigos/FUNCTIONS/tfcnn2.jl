#Segundo modelo para prever caracteres
using DataFrames
using HDF5, JLD
using Distributions
using TensorFlow

xtrain, ytrain = load("fold1xwb.jld")["xtrain"],load("fold1ywb.jld")["ytrain"];

ytrain2 = zeros(0,26);
xtrain2 = zeros(0,1024);

for i = 1:25
  a = find(ytrain[:,i] .== 1.0)
  ytrain2 = [ytrain2; ytrain[a,1:26]]
  xtrain2 = [xtrain2; xtrain[a,:]]
end

for i = 1:1000
  c = zeros(26)
  c[26] = 1.0
  ytrain2 = [ytrain2; c']
end

b = sample(1:size(xtrain,1),1000,replace = false);
xtrain2 = [xtrain2; xtrain[b,:]];

random_vector = sample(1:size(ytrain2,1),size(ytrain2,1),replace = false);
ytrain2 = ytrain2[random_vector,:];
xtrain2 = xtrain2[random_vector,:];


function next_batch2(xtrain, ytrain, batch_size,i)
  n = size(xtrain,1)
  if i < 105
    vector = collect((i*batch_size):(i*batch_size+batch_size))
    return xtrain[vector,:], ytrain[vector,:]
  elseif i == 105
    a = 104*batch_size + batch_size + 1
    vector = collect((():n)
    return xtrain[vector,:], ytrain[vector,:]
  end
end

session2 = Session(Graph())

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

x2 = placeholder(Float32)
y2_ = placeholder(Float32)

W_conv12 = weight_variable([3, 3, 1, 32])
b_conv12 = bias_variable([32])

x_image2 = reshape(x2, [-1, 32, 32, 1])

h_conv12 = nn.relu(conv2d(x_image2, W_conv12) + b_conv12)
h_pool12 = max_pool_2x2(h_conv12)

W_conv22 = weight_variable([3, 3, 32, 64])
b_conv22 = bias_variable([64])

h_conv22 = nn.relu(conv2d(h_pool12, W_conv22) + b_conv22)
h_pool22 = max_pool_2x2(h_conv22)

W_fc12 = weight_variable([8*8*64, 1024])
b_fc12 = bias_variable([1024])

h_pool2_flat2 = reshape(h_pool22, [-1, 8*8*64])
h_fc12 = nn.relu(h_pool2_flat2 * W_fc12 + b_fc12)

keep_prob2 = placeholder(Float32)
h_fc1_drop2 = nn.dropout(h_fc12, keep_prob2)

W_fc22 = weight_variable([1024, 26])
b_fc22 = bias_variable([26])

y_conv2 = nn.softmax(h_fc1_drop2 * W_fc22 + b_fc22)

cross_entropy2 = reduce_mean(-reduce_sum(y2_.*log(y_conv2+1e-10), reduction_indices=[2]))

train_step2 = train.minimize(train.AdamOptimizer(.001), cross_entropy2)

correct_prediction2 = indmax(y_conv2, 2) .== indmax(y2_, 2)

accuracy2 = reduce_mean(cast(correct_prediction2, Float32))

run(session2, initialize_all_variables())

for j = 1:5
  for i in 1:104
    batch = next_batch2(xtrain,ytrain2,27,i)
    if i%25 == 0
      train_accuracy = run(session2, accuracy2, Dict(x2=>batch[1], y2_=>batch[2], keep_prob2=>1.0))
      info("step $((j-1)*150+i*j), training accuracy $train_accuracy")
    end
    run(session2, train_step2, Dict(x2=>batch[1], y2_=>batch[2], keep_prob2=>.5))
  end
end

xtest = load("fold1txwb.jld")["xtest"];
ytest = load("fold1tywb.jld")["ytest"];

ytest2 = zeros(size(xtest,1),26);

for i = 1:size(ytest2,1)
  a = find(ytest[i,:].==maximum(ytest[i,:]))[1]
  if a <= 25
    ytest2[i,a] = 1.0
  else
    ytest2[i,26] = 1.0
  end
end

acuracia = 0;

for i = 1:50
  batch = next_batch2(xtest,ytest2,250,i)
  test_accuracy = run(session2, accuracy2, Dict(x2=>batch[1], y2_=>batch[2], keep_prob2=>1.0))
  info("test accuracy $test_accuracy")
  acuracia += test_accuracy
end
acuracia = acuracia/32;
println("Accuracy $acuracia")
