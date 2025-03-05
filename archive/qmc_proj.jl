#= using LinearAlgebra

using Statistics
using DelimitedFiles

n = 4
graph_name = "Shastry-Sutherland_N$n"
graph_path = "./test/testdata/graphdata/$graph_name.dat"
edges_raw = readdlm(graph_path, Int)
edge_set = [tuple(sort(edges_raw[i, 1:2])..., 1.0) for i in axes(edges_raw, 1)]

g = load_weighted_graph("./data/Shastry-Sutherland.txt", graph_name * "_EQW")

tmp_g = SimpleWeightedGraph(n)
for e in edge_set
	if has_edge(tmp_g, e[1], e[2])
		tmp_g.weights[e[1], e[2]] += e[3]
	else
		add_edge!(tmp_g, e[1], e[2], e[3])
	end
end
@show edge_set
@show tmp_g.weights


optimize!(model)

@show objective_value(model);

#=
X_sol = value.(X)
correlations = Dict(var => MM[pos[1], pos[2]] for (var, pos) in var_dict_constr_pos)
moments_free = Dict(var => MM[pos[1], pos[2]] for (var, pos) in var_dict_free_pos)
moments = merge(correlations, moments_free)
=#

=#
