using LinearAlgebra, SparseArrays, JuMP, MosekTools, DelimitedFiles

# Pauli matrices
const I = sparse(ComplexF64[1 0; 0 1])
const X = sparse(ComplexF64[0 1; 1 0])
const Y = sparse(ComplexF64[0 -im; im 0])
const Z = sparse(ComplexF64[1 0; 0 -1])

# Sparse Kronecker product
sparse_kron(A, B) = kron(A, B)

# Operator acting on specific qubits
function op_q(op::Tuple, edge::Tuple, n::Int)
    tprod = [I for _ in 1:n]
    for (idx, qubit) in enumerate(edge)
        tprod[qubit+1] = op[idx]  # Julia is 1-based indexing
    end
    return reduce(sparse_kron, tprod)
end

# Singlet operator
singlet(edge::Tuple, n::Int) = op_q((X,X), edge, n) + op_q((Y,Y), edge, n) + op_q((Z,Z), edge, n)

# Exact diagonalization for quantum maxcut
function qmc_exact_weighted(n::Int, edges::Vector{Tuple{Int,Int,Float64}})
    H = spzeros(ComplexF64, 2^n, 2^n)
    weight_sum = 0.0
    for (i,j,w) in edges
        H += w * singlet((i,j), n)
        weight_sum += w
    end
    energy_min, _ = eigs(H, nev=1, which=:SR)
    energy_max = (weight_sum - real(energy_min[1])) / 4
    return energy_max
end

# Multiplication algebra for projectors
function proj_mult(a::Tuple, b::Tuple)
    if a == (0,0) && b != (0,0)
        return ([1.0], [b], 2)
    elseif a != (0,0) && b == (0,0)
        return ([1.0], [a], 2)
    elseif a == b
        return ([1.0], [a], 2)
    else
        a_set, b_set = Set(a), Set(b)
        c = intersect(a_set, b_set)
        u = union(a_set, b_set)
        if isempty(c)
            return ([1.0], [(a,b)], 0)
        elseif length(c) == 2
            return ([1.0], [a], 2)
        elseif length(c) == 1
            e = first(c)
            delete!(u, e)
            u_sorted = sort(collect(u))
            return ([1/4, 1/4, -1/4], [a, b, tuple(u_sorted...)], 1)
        end
    end
end

# Data preparation for SDP
function get_data_weighted(n::Int, edges::Vector{Tuple{Int,Int,Float64}})
    htuples = [(0,0); [(i,j) for i in 0:n-2 for j in i+1:n-1]]
    n_htuples = length(htuples)

    var_dict_constr_pos = Dict(h => (i, 0) for (i,h) in enumerate(htuples))
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
    for (i,j,w) in edges
        pos = var_dict_constr_pos[(i,j)]
        push!(C[1], pos[1])
        push!(C[2], pos[2])
        push!(C[3], w)
    end

    X_dim = div(n*(n-1),2) + 1
    return A, C, X_dim, var_dict_constr_pos, var_dict_free_pos
end

# Solve SDP using JuMP and Mosek
function solve_main_weighted(n::Int, edges::Vector{Tuple{Int,Int,Float64}})
    A, C, X_dim, _, _ = get_data_weighted(n, edges)

    model = Model(Mosek.Optimizer)
    set_silent(model)

    @variable(model, X[1:X_dim,1:X_dim], PSD)
    @objective(model, Max, sum(C[3][k]*X[C[1][k],C[2][k]] for k in eachindex(C[3])))

    @constraint(model, X[1,1] == 1.0)
    for a in values(A)
        @constraint(model, sum(a[3][k]*X[a[1][k],a[2][k]] for k in eachindex(a[3])) == 0.0)
    end

    optimize!(model)
    obj_val = objective_value(model)
    MM = value.(X)

    return obj_val, MM
end

# Main execution
function main()
    n = 48
    graph_name = "Majumdar-Ghosh_N$n"
    graph_path = "./graph_data/$graph_name.dat"
    edges_raw = readdlm(graph_path, Int)
    edges = [(min(e[1],e[2]), max(e[1],e[2]), 1.0) for e in eachrow(edges_raw)]

    dir_result = "./$graph_name/"
    isdir(dir_result) || mkpath(dir_result)

    t_start = time()
    E_proj, moment_matrix = solve_main_weighted(n, edges)
    t_proj = round(time() - t_start, digits=2)

    println("Start ED")
    E_exact = qmc_exact_weighted(n, edges)
    t_exact = round(time() - t_start - t_proj, digits=2)
    E_diff = abs(E_exact - E_proj)

    println("Exact energy: E = $E_exact")
    println("Approximated energy: L = $E_proj")
    println("Difference: D = $E_diff")
    println("Time elapsed proj: $t_proj s")
    println("Time elapsed exact: $t_exact s")

    open("$dir_result$(graph_name)_stats.txt", "w") do f
        write(f, "$graph_name $E_exact $E_proj $E_diff\n")
    end
    open("$dir_result$(graph_name)_mm.npy", "w") do f
        write(f, moment_matrix)
    end
end

main()