using Mosek, MosekTools
using JuMP

using DelimitedFiles

n = 32
graph_name = "Majumdar-Ghosh_N$n"
graph_path = "./graph_data/$graph_name.dat"
edges_raw = readdlm(graph_path, Int)
edge_set = [tuple(sort(edges_raw[i,1:2])..., Float64(edges_raw[i,3])) for i in axes(edges_raw, 1)]

model = Model(Mosek.Optimizer)
set_silent(model)

@show edge_set

# Data preparation for SDP
function get_data_weighted(n::Int, edges::Vector{Tuple{Int,Int,Float64}})
    htuples = [(1,1); [(i,j) for i in 1:n-1 for j in i+1:n]]
    n_htuples = length(htuples)

    var_dict_constr_pos = Dict(h => (i, 1) for (i,h) in enumerate(htuples))
    var_dict_free_pos = Dict{Tuple,Tuple}()
    A = Dict{Tuple,Tuple{Vector{Int},Vector{Int},Vector{Float64}}}()

    for i in 1:n_htuples, j in i:n_htuples
        a, b = htuples[i], htuples[j]
        if i == 1 && j >= 2
            A[(a,b)] = ([j,j], [i,j], [1.0,-1.0])
        elseif i > 1 && j > i
            a_set, b_set = Set(a), Set(b)
            c = intersect(a_set, b_set)
            u = union(a_set, b_set)
            if isempty(c)
                var_dict_free_pos[(a,b)] = (j,i)
            elseif length(c) == 1
                e = first(c)
                delete!(u, e)
                u_sorted = sort(collect(u))
                pos_a, pos_b, pos_c = var_dict_constr_pos[a], var_dict_constr_pos[b], var_dict_constr_pos[tuple(u_sorted...)]
                A[(a,b)] = ([j,pos_a[1],pos_b[1],pos_c[1]], [i,pos_a[2],pos_b[2],pos_c[2]], [-1.0,1/4,1/4,-1/4])
            end
        end
    end

    C = (Int[], Int[], Float64[])
    for (i, j, w) in edges
        pos = var_dict_constr_pos[(i + 1, j + 1)]
        push!(C[1], pos[1])
        push!(C[2], pos[2])
        push!(C[3], w)
    end

    X_dim = div(n*(n-1),2) + 1
    return A, C, X_dim, var_dict_constr_pos, var_dict_free_pos
end

A, C, X_dim, var_dict_constr_pos, var_dict_free_pos = get_data_weighted(n, edge_set)

@show A |> collect |> size
@show C |> collect |> size
@show X_dim
@show var_dict_constr_pos |> collect |> size
@show var_dict_free_pos |> collect |> size

@variable(model, X[1:X_dim,1:X_dim], PSD)
@objective(model, Max, sum(C[3][k] * X[C[1][k], C[2][k] + 1] for k in 1:length(C[3])))
@constraint(model, X[1,1] == 1.0)

@show A
for a in values(A)
    @constraint(model, sum(a[3][k]*X[a[1][k],a[2][k]] for k in eachindex(a[3])) == 0.0)
end

optimize!(model)
obj_val = objective_value(model)
MM = value.(X)

for a in values(A)
    @show a[1], a[2], a[3]
end

@show A