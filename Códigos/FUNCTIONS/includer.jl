include("separ.jl")
include("resizing.jl")
include("imgmatrix.jl")
include("getmatrix.jl")

#Concatena as colunas da matriz em um vetor
function concatenatematrix(matriz::Matrix)
  m,n = size(matriz)
  S = zeros(0)
  for i = 1:n
    S = vcat(S,matriz[:,i])
  end
  return S
end

imgr = imgmatrix("TesteMate.png")

imgr = imgmatrix("TesteUFPR.png")

listao = recpa(imgr);
count = 0;
S = zeros(Int64,0);
for i = 1:size(listao,1)
  if !(listao[i,3] in S)
    count += 1
    S = vcat(S,listao[i,3])
  end
end
for i = 1:length(S)
  list = indexoflist(listao,i);
  list = getmatrix(list,imgr);
  resize = resizor(list,32,32)
  resized = zeros(32,32);
  for i = 1:32,j=1:32
    if resize[i,j] < 0.3
      resized[i,j]=0.0
    else
      resized[i,j]=1.0
    end
  end
  imshow(resized)
  res = concatenatematrix(resized)
  feed_dict = Dict(x=>res,keep_prob=>1.0)
  result = run(session,y_conv,feed_dict)
  b = find(result .== maximum(result))[1]
  println("O símbolo correspondente da tabela é o símbolo $(symbola[:latex][b]).")
end
