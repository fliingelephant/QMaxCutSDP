using LinearAlgebra
using SparseArrays
using JuMP
using MosekTools
using Statistics
using DelimitedFiles

function get_data_weighted(n, edges)
	# Get all the h tuples (i,j), i < j, and (1,1)
	# Note: Adjusted from (0,0) to (1,1) for Julia's 1-based indexing
	htuples = [(1, 1)]
	for i in 1:n
		for j in i+1:n
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
	for (i, j, w) in edges
		pos = var_dict_constr_pos[(i + 1, j + 1)]
		push!(C[1], pos[1])
		push!(C[2], pos[2])
		push!(C[3], w)
	end

	# Dimension of moment matrix M
	X_dim = Int(n * (n - 1) / 2 + 1)

	return A, C, X_dim, var_dict_constr_pos, var_dict_free_pos
end

function solve_main_weighted(n, edges)
	"""
	Main function for level-1 projector SDP for QMAXCUT. 

	Edges are weighted, i.e., edges = [(i_1, j_1, w_{i_1j_1}),...,(i_s, j_s, w_{i_sj_s})].
	"""

	A, C, X_dim, var_dict_constr_pos, var_dict_free_pos = get_data_weighted(n, edges)

	model = Model(Mosek.Optimizer)

	# Setting up the variables
	@variable(model, X[1:X_dim, 1:X_dim], PSD)

	# Objective
	C_matrix = sparse(C[1], C[2], C[3], X_dim, X_dim)
	@objective(model, Max, dot(C_matrix, X))

	# Constraints
	@constraint(model, X[1, 1] == 1.0)

	for a in values(A)
		B = sparse(a[1], a[2], a[3], X_dim, X_dim)
		@constraint(model, dot(B, X) == 0.0)
	end

	# Solve
	optimize!(model)

	# We could extract correlations and moments here if needed
	# correlations = Dict(var => MM[pos[1], pos[2]] for (var, pos) in var_dict_constr_pos)
	# moments_free = Dict(var => MM[pos[1], pos[2]] for (var, pos) in var_dict_free_pos)
	# moments = merge(correlations, moments_free)

	return objective_value(model), value.(X)
end

n = 32
graph_name = "Majumdar-Ghosh_N$n"
graph_path = "./graph_data/$graph_name.dat"
edges_raw = readdlm(graph_path, Int)
edge_set = [tuple(sort(edges_raw[i, 1:2])..., 1.0) for i in axes(edges_raw, 1)]

model = Model(Mosek.Optimizer)

A, C, X_dim, var_dict_constr_pos, var_dict_free_pos = get_data_weighted(n, edge_set)

@variable(model, X[1:X_dim, 1:X_dim], PSD)

C_matrix = sparse(C[1], C[2], C[3], X_dim, X_dim)
@objective(model, Max, dot(C_matrix, X))

@constraint(model, X[1, 1] == 1.0) # zeroth-order moment should be 1

[@constraint(model,
    sum([coefficient * X[column_index, row_index] 
        for (column_index, row_index, coefficient) in zip(a[1], a[2], a[3])]
    ) == 0.0)
for a in values(A)]

optimize!(model)

objective_value = objective_value(model)

X_sol = value.(X)

correlations = Dict(var => MM[pos[1], pos[2]] for (var, pos) in var_dict_constr_pos)
moments_free = Dict(var => MM[pos[1], pos[2]] for (var, pos) in var_dict_free_pos)
moments = merge(correlations, moments_free)