#transform your string labels into a 0-1 matrix, where each line represents one
#coordinate of the input vector. The position of the one determines from which
#class the data is.

function convertOneHot(y::Any, number_of_classes::Int64;vector=true)
  if vector == true
    a = length(y)
    S = []
    for i = 1:a
      if !(y[i] in S)
        S = [S;y[i]]
      end
    end
    b = length(S)
    d = zeros(Int,a)
    for j = 1:b
      c = find(y .== S[j])
      d[c] = j
    end
    return d
  else
    a = length(y)
    S = []
    for i = 1:a
      if !(y[i] in S)
        S = [S;y[i]]
      end
    end
    b = number_of_classes
    d = zeros(Int,a,b)
    for j = 1:length(S)
      c = find(y .== S[j])
      d[c,j] = 1
    end
    return d
  end
end
