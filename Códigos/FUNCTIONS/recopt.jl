function rec(A::Matrix)
  m,n = size(A)
  B = zeros(m,n)
  list = Array{Float64}(0,2)
  counter_list = []
  counter = 0
  contando = 0
  for j = 1:n, i = 1:m
    if A[i,j] != 0.0
      w = sum(all(list .== [i j], 2))
      if w != 0
        a = find(all(list .== [i j],2))[1]
        for k in [-1,1]
          if A[i+k,j] != 0.0
            if sum(all(list .== [(i+k) j],2)) == 0
              list = vcat(list, [(i+k) j])
              counter_list = vcat(counter_list, counter_list[a])
              B[i+k,j] = counter_list[a]
            end
          end
          if A[i,j+k] != 0.0
            if sum(all(list .== [i (j+k)],2)) == 0
              list = vcat(list, [i (j+k)])
              counter_list = vcat(counter_list, counter_list[a])
              B[i,j+k] = counter_list[a]
            end
          end
          if A[i+k,j+k] != 0.0
            if sum(all(list .== [(i-1) (j+1)],2)) == 0
              list = vcat(list, [(i-1) (j+1)])
              counter_list = vcat(counter_list, counter_list[a])
              B[i+k,j+k] = counter_list[a]
            end
          end
          if A[i+k,j-k] != 0.0
            if sum(all(list .== [(i+1) (j-1)],2)) == 0
              list = vcat(list, [(i+1) (j-1)])
              counter_list = vcat(counter_list, counter_list[a])
              B[i+k,j-k] = counter_list[a]
            end
          end
        end
      else
        if sum(all(list[:,2].== j],2))
          list = vcat(list, [i j])
          a = find(all(list .== [i j],2))[1]
          for k in [-1,1]
            if A[i+k,j] != 0.0
              if sum(all(list .== [(i+k) j],2)) == 0
                list = vcat(list, [(i+k) j])
                counter_list = vcat(counter_list, counter_list[a])
                B[i+k,j] = counter_list[a]
              end
            end
            if A[i,j+k] != 0.0
              if sum(all(list .== [i (j+k)],2)) == 0
                list = vcat(list, [i (j+k)])
                counter_list = vcat(counter_list, counter_list[a])
                B[i,j+k] = counter_list[a]
              end
            end
            if A[i+k,j+k] != 0.0
              if sum(all(list .== [(i-1) (j+1)],2)) == 0
                list = vcat(list, [(i-1) (j+1)])
                counter_list = vcat(counter_list, counter_list[a])
                B[i+k,j+k] = counter_list[a]
              end
            end
            if A[i+k,j-k] != 0.0
              if sum(all(list .== [(i+1) (j-1)],2)) == 0
                list = vcat(list, [(i+1) (j-1)])
                counter_list = vcat(counter_list, counter_list[a])
                B[i+k,j-k] = counter_list[a]
              end
            end
          end
        list = vcat(list, [i j])
        counter += 1
        counter_list = vcat(counter_list, counter)
        B[i,j] = counter
        for k in [-1,1]
          if A[i+k,j] != 0.0
            list = vcat(list, [(i+k) j])
            counter_list = vcat(counter_list, counter)
            B[i+k,j] = counter
          end
          if A[i,j+k] != 0.0
            list = vcat(list, [i (j+k)])
            counter_list = vcat(counter_list, counter)
            B[i,j+k] = counter
          end
          if A[i+k,j+k] != 0.0
            list = vcat(list, [(i+k) (j+k)])
            counter_list = vcat(counter_list, counter)
            B[i+k,j+k] = counter
          end
          if A[i+k,j-k] != 0.0
            list = vcat(list, [(i+k) (j-k)])
            counter_list = vcat(counter_list, counter)
            B[i+k,j-k] = counter
          end
        end
      end
    end
  end
  listao = [list counter_list]
  listao = convert(Array{Int64}, listao)
  return listao
end
