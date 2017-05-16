#convert images to matrix
using Images, ImageView

function imgmatrix(path)
  img = load("$path")
  img = convert(ImageMeta{Gray}, img)
  imgr = convert(Array{Float64}, rawview(channelview(img)))
  imgr = imgr./255
  return abs(1.-imgr)
end
