function inlist(u,v,D)
  return any((D[:,1] .== u) & (D[:,2] .== v))
end

function findNeigh(i,j,A,D; r = 1)
  L = []
  for p = -r:r, q = -r:r
    if p == q == 0
      continue
    end
    if A[i+p,j+q] != 0.0 && !inlist(i+p,j+q,D)
      push!(L, (i+p,j+q))
    end
  end
  return L
end

function recpa(A::Matrix)
  m,n = size(A)
  D = zeros(Int,0,3)
  k = 0
  for i = 1:m, j = 1:n
    if A[i,j] != 0.0 && !inlist(i,j,D)
      k += 1
      S = [(i,j)]
      while !isempty(S)
        (p,q) = pop!(S)
        D = [D; [p,q,k]']
        L = findNeigh(p,q,A,D)
        for (u,v) in L
          if !((u,v) in S) && !inlist(u,v,D)
            push!(S, (u,v))
          end
        end
      end
    end
  end
  return D
end

function indexoflist(listao, i)
  list = zeros(Int,0,2)
  for j = 1:size(listao,1)
    if listao[j,3] == i
      list = vcat(list, listao[j,1:2]')
    end
  end
  return list
end
