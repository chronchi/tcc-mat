#takes a matrix, turns into an image, resizes it
#and then converts back to the [0,1] gray scale matrix
#to run the predictor

using Images, ImageView #pacotes a serem usados

#(m,n) tamanho para resize
function resizor(A::Matrix,m::Int64,n::Int64)
  #normalização para a escala [0,1]
  #converte matriz para imagem na escala cinza
  img = colorview(Gray,A)
  #reajuste imagem para (m,n)
  resized = Images.imresize(img, (m,n))
  #converte imagem para matriz
  imgr = convert(Array{Float64}, rawview(channelview(resized)))
  #retorna matriz de numeros da imagem reajustada
  return imgr
end
