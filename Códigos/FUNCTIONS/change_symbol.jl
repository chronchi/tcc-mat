function changesymb(symbol,true_symbol)
  m = length(symbol)
  counter = 0
  list = zeros(m)
  for i = 1:m
    a = find(true_symbol .== symbol[i])[1]
    list[i] = a
  end
  list = convert(Array{Int64}, list)
  return list
end
