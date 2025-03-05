function get_data(g::SimpleWeightedGraph)
    n = nv(g)

    htuples = [(1, 1)]
    for i in 1:n
        for j in (i + 1):n
            push!(htuples, (i, j))
        end
    end

    n_htuples = length(htuples)
    # Dictionary storing the positions of the constraint variables, i.e., all the h_{ij}'s.
    var_dict_constr_pos = Dict(h => (i, 1) for (i, h) in enumerate(htuples))

    # Dictionary storing the positions of the free variables, i.e., all the h_{ij}h_{kl}, 
    # where there's no overlap between {i,j} and {k, l}.
    var_dict_free_pos = Dict()

    # A stores all sparse matrices that specify the equality constraints in moment matrix M such that Tr(M A_i) = 0
    A = Dict()

    for i in 1:n_htuples
        for j in i:n_htuples
            a, b = htuples[i], htuples[j]
            if i == 1 && j >= 2
                A[(a, b)] = (Int[], Int[], Float64[])
                push!(A[(a, b)][1], j)
                push!(A[(a, b)][1], j)
                push!(A[(a, b)][2], i)
                push!(A[(a, b)][2], j)
                push!(A[(a, b)][3], 1)
                push!(A[(a, b)][3], -1)
            elseif i > 1 && j > i
                a_set = Set(a)
                b_set = Set(b)
                c = intersect(a_set, b_set)
                u = union(a_set, b_set)
                if length(c) == 0
                    var_dict_free_pos[(a, b)] = (j, i)
                elseif length(c) == 1
                    # get the single element from set c
                    e = first(c)
                    delete!(u, e)
                    pos_a = var_dict_constr_pos[a]
                    pos_b = var_dict_constr_pos[b]
                    pos_c = var_dict_constr_pos[Tuple(sort([i for i in u]))]
                    A[(a, b)] = (Int[], Int[], Float64[])
                    push!(A[(a, b)][1], j, pos_a[1], pos_b[1], pos_c[1])
                    push!(A[(a, b)][2], i, pos_a[2], pos_b[2], pos_c[2])
                    push!(A[(a, b)][3], -1, 1 / 4, 1 / 4, -1 / 4)
                end
            end
        end
    end

    # Obtain matrix C in sparse format
    C = [Int[], Int[], Float64[]]
    for e in edges(g)
        pos = var_dict_constr_pos[(e.src, e.dst)]
        push!(C[1], pos[1])
        push!(C[2], pos[2])
        push!(C[3], e.weight)
    end

    # Dimension of moment matrix M
    X_dim = Int(n * (n - 1) / 2 + 1)

    return A, C, X_dim, var_dict_constr_pos, var_dict_free_pos
end
