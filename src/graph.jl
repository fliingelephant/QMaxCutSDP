function load_graph(graph_name::String)
end

using DelimitedFiles
using SimpleWeightedGraphs
using Graphs
n = 32
graph_name = "Majumdar-Ghosh_N$n"
graph_path = "./graph_data/$graph_name.dat"
edges_raw = readdlm(graph_path, Int)

g = SimpleWeightedGraph(32)

for e in eachrow(edges_raw)
    @show e
    add_edge!(g, e[1], e[2], e[3])
end

saveswg("./data/graph.txt", Dict(graph_name => g), LGFormat())