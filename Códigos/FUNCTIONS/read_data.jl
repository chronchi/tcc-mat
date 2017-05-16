#labels: train data using dataframes
#imageSize: size of images, which is 32x32 = 1024
#path: path of images

function concatenatematrix(matriz::Matrix)
  m,n = size(matriz)
  S = zeros(0)
  for i = 1:n
    S = vcat(S,matriz[:,i])
  end
  return S
end


using Images

function read_data(imageSize, path, symbol)
  numb_labels = length(symbol)

  #initialize x matrix
  x = zeros(numb_labels, imageSize)

  for i in 1:numb_labels
    nameFile = "$(path)/$(symbol[i])"
    img = load("$nameFile")
    img = convert(ImageMeta{Gray}, img)
    imgr = convert(Array{Float64}, rawview(channelview(img)))
    imgr = abs(1.- (imgr./255))
    if i%10000 == 0
      info("imagem $i")
    end 
    x[i,:] = concatenatematrix(imgr)
  end
  return x
end
