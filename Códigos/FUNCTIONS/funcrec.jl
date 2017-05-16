function rec(A::Matrix)
  m,n = size(A)
  B = zeros(m,n)
  list = Array{Float64}(0,2)
  counter_list = []
  counter = 0
  contando = 0
  for i = 1:m, j = 1:n
    if A[i,j] == 0.0
      continue
    end
    w = sum(all(list .== [i j], 2))
    if w != 0
      a = find(all(list .== [i j],2))[1]
      if A[i-1,j] != 0.0
        if sum(all(list .== [(i-1) j],2)) == 0
          list = vcat(list, [(i-1) j])
          counter_list = vcat(counter_list, counter_list[a])
          B[i-1,j] = counter_list[a]
        end
      end
      if A[i+1,j] != 0.0
        if sum(all(list .== [(i+1) j],2)) == 0
          list = vcat(list, [(i+1) j])
          counter_list = vcat(counter_list, counter_list[a])
          B[i+1,j] = counter_list[a]
        end
      end
      if A[i,j-1] != 0.0
        if sum(all(list .== [i (j-1)],2)) == 0
          list = vcat(list, [i (j-1)])
          counter_list = vcat(counter_list, counter_list[a])
          B[i,j-1] = counter_list[a]
        end
      end
      if A[i,j+1] != 0.0
        if sum(all(list .== [i (j+1)],2)) == 0
          list = vcat(list, [i (j+1)])
          counter_list = vcat(counter_list, counter_list[a])
          B[i,j+1] = counter_list[a]
        end
      end
      if A[i-1,j+1] != 0.0
        if sum(all(list .== [(i-1) (j+1)],2)) == 0
          list = vcat(list, [(i-1) (j+1)])
          counter_list = vcat(counter_list, counter_list[a])
          B[i-1,j+1] = counter_list[a]
        end
      end
      if A[i-1,j-1] != 0.0
        if sum(all(list .== [(i-1) (j-1)],2)) == 0
          list = vcat(list, [(i-1) (j-1)])
          counter_list = vcat(counter_list, counter_list[a])
          B[i-1,j-1] = counter_list[a]
        end
      end
      if A[i+1,j-1] != 0.0
        if sum(all(list .== [(i+1) (j-1)],2)) == 0
          list = vcat(list, [(i+1) (j-1)])
          counter_list = vcat(counter_list, counter_list[a])
          B[i+1,j-1] = counter_list[a]
        end
      end
      if A[i+1,j+1] != 0.0
        if sum(all(list .== [(i+1) (j+1)],2)) == 0
          list = vcat(list, [(i+1) (j+1)])
          counter_list = vcat(counter_list, counter_list[a])
          B[i+1,j+1] = counter_list[a]
        end
      end
    else
      list = vcat(list, [i j])
      counter += 1
      counter_list = vcat(counter_list, counter)
      B[i,j] = counter
      if A[i-1,j] != 0.0
        list = vcat(list, [(i-1) j])
        counter_list = vcat(counter_list, counter)
        B[i-1,j] = counter
      end
      if A[i,j-1] != 0.0
        list = vcat(list, [i (j-1)])
        counter_list = vcat(counter_list, counter)
        B[i,j-1] = counter
      end
      if A[i,j+1] != 0.0
        list = vcat(list, [i (j+1)])
        counter_list = vcat(counter_list, counter)
        B[i,j+1] = counter
      end
      if A[i+1,j] != 0.0
        list = vcat(list, [(i+1) j])
        counter_list = vcat(counter_list, counter)
        B[i+1,j] = counter
      end
      if A[i-1,j-1] != 0.0
        list = vcat(list, [(i-1) (j-1)])
        counter_list = vcat(counter_list, counter)
        B[i-1,j-1] = counter
      end
      if A[i-1,j+1] != 0.0
        list = vcat(list, [(i-1) (j+1)])
        counter_list = vcat(counter_list, counter)
        B[i-1,j+1] = counter
      end
      if A[i+1,j-1] != 0.0
        list = vcat(list, [(i+1) (j-1)])
        counter_list = vcat(counter_list, counter)
        B[i+1,j-1] = counter
      end
      if A[i+1,j+1] != 0.0
        list = vcat(list, [(i+1) (j+1)])
        counter_list = vcat(counter_list, counter)
        B[i+1,j+1] = counter
      end
    end
  end
  listao = [list counter_list]
  listao = convert(Array{Int64}, listao)
  return listao
end

function sortcord(list::Matrix)
  n = size(list,1)
  counter = 0
  lista1 = Array{Int64}(0,2)
  while counter <= n
    a = minimum(list[:,1])
    d = find(all(list[:,1] .== a,2))
    e = []
    for j in d
      e = vcat(e,list[j,2])
    end
    b = minimum(e)
    lista1 = vcat(lista1, [a b])
    c = find(all(list .== [a b],2))
    if length(c) == 0
      break
    end
    counter += 1
    if counter == n
      break
    end
    list = list[1:end .!= c[1],:]
  end
  #organizar list recursivamente, removendo
  #seus elementos e adicionando-os em outra lista
  return lista1
end

function numerolista(list::Matrix)
  n = size(list,1)
  a = 0
  quant = []
  for i in 1:n
    if list[i,3] in quant
      continue
    end
    a += 1
    quant = vcat(quant, a)
  end
  return quant
end
