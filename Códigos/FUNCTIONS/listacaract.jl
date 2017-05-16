#primeiro obtem a imagem
#separa a imagem com rec()
#obtem as matrizes de cada imagem com getmatrix
#para cada imagem getmatrix
  #reajuste de matrix para 32x32
  #predição da imagem
#end

for i in quant #quant é a lista gerada por numero lista
  #talvez crie arquivo para salvar a predição
  list = Array{Int64}(0,2)
  for j = 1:size(listao,1)
    if listao[j,3] == i
      list = vcat(list,lista[j,1:2])
    end
  end
  list = sortcord(list)
  listm = getmatrix(list)
  #Aqui se aplica o modelo para prever que caracter é com listm
  #println("caracter") ou salva em uma lista num arquivo txt/csv
end
