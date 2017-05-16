#using HDF5, JLD
#using Distributions
#using TensorFlow
#using DataFrames

#xtrain = load("xtrain.jld")["xtrain"]
#y_train = readtable("train.csv")
#ytrain = convert(Array{Float64}, y_train[:symbol_id])


#ytrainnew = changesymb(convert(Array{Int64}, ytrain), convert(Array{Int64}, symbols))
#ytrainmatrix = zeros(size(ytrain), 369);
#for i = 1:length(ytrain)
#  ytrainmatrix[i,ytrainnew[i]] = 1
#end
Ytrain = ytrainmatrix;
Xtrain = xtrain;


m,n=size(Xtrain)
classes = 369

sess = Session(Graph())
X = placeholder(Float64)
Y_obs = placeholder(Float64)

variable_scope("logisitic_model", initializer=Normal(0, .001)) do
    global W = get_variable("weights", [n, classes], Float64)
    global B = get_variable("bias", [classes], Float64)
end

Y=nn.softmax(X*W + B)
Loss = -reduce_sum(log(Y).*Y_obs)
optimizer = train.AdamOptimizer()
minimize_op = train.minimize(optimizer, Loss)
saver = train.Saver()
# Run training
run(sess, initialize_all_variables())
checkpoint_path = mktempdir()
info("Checkpoint files saved in $checkpoint_path")
for epoch in 1:2
    cur_loss, _ = run(sess, vcat(Loss, minimize_op), Dict(X=>Xtrain, Y_obs=>Ytrain))
    println(@sprintf("Current loss is %.2f.", cur_loss))
    train.save(saver, sess, joinpath(checkpoint_path, "logistic"), global_step=epoch)
end
