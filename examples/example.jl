using QMaxCutSDP

using JuMP

data_path = "./data/"

n = 36
#graph_type = "Majumdar-Ghosh"
graph_type = "Shastry-Sutherland"
graph_name = graph_type * "_N$n"

g = load_weighted_graph(data_path * graph_type * ".txt", graph_name * "_EQW")

A, C, X_dim, var_dict_constr_pos, var_dict_free_pos = get_data(g);

model = make_sdp(A, C, X_dim, var_dict_constr_pos, var_dict_free_pos);

optimize!(model)

@show objective_value(model)
