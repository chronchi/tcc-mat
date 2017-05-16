using DataFrames

function load_data(originalPath, imgSize)
  train_total = readtable("$originalPath/train.csv")
  test_total = readtable("$originalPath/test.csv")
  m = length(train_total[:path])
  X = zeros(m,imgSize)
  y = zeros(m)
  train = train_total[:path]
  test = test_total[:path]
  for i = 1:m
    if i%1000 == 0
      println(i)
    end
    nameFile = "$originalPath/$(train[i])"
    img = load(nameFile)
    imga = convert(Array{Float64}, rawview(channelview(img)))
    X[i,:] = imga
    y[i] = train_total[:symbol_id][i]
  end
  return X,y
end
