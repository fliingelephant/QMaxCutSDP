using DelimitedFiles
using QMaxCutSDP
using Test
using SimpleWeightedGraphs

@testset "Graph persistence" begin
    data_path = "./data/"
    Jun_data_path = "./test/testdata/graphdata/"

    for n in [16, 32, 48, 64]
        graph_type = "Majumdar-Ghosh"
        graph_name = graph_type * "_N$n"

        graph_dat_path = Jun_data_path * "$graph_name.dat"
        edges_raw = readdlm(graph_dat_path, Int)
        g1 = SimpleWeightedGraph(n)
        for e in eachrow(edges_raw)
            SimpleWeightedGraphs.add_edge!(g1, e[1] + 1, e[2] + 1, e[3])
        end

        g2 = load_weighted_graph(data_path * graph_type * ".txt", graph_name)

        @test g1 == g2
    end

    for n in [4, 16, 36, 64]
        graph_type = "Shastry-Sutherland"
        graph_name = graph_type * "_N$n"

        graph_dat_path = Jun_data_path * "$graph_name.dat"
        edges_raw = readdlm(graph_dat_path, Int)
        g1 = SimpleWeightedGraph(n)
        for e in eachrow(edges_raw)
            SimpleWeightedGraphs.add_edge!(g1, e[1] + 1, e[2] + 1, e[3])
        end

        g2 = load_weighted_graph(data_path * graph_type * ".txt", graph_name)

        @test g1 == g2
    end
end

# Equal weight models
@testset "Graph persistence" begin
    data_path = "./data/"
    Jun_data_path = "./test/testdata/graphdata/"

    for n in [16, 32, 48, 64]
        graph_type = "Majumdar-Ghosh"
        graph_name = graph_type * "_N$n"

        graph_dat_path = Jun_data_path * "$graph_name.dat"
        edges_raw = readdlm(graph_dat_path, Int)
        g1 = SimpleWeightedGraph(n)
        for e in eachrow(edges_raw)
            SimpleWeightedGraphs.add_edge!(g1, e[1] + 1, e[2] + 1, 1.0)
        end

        g2 = load_weighted_graph(data_path * graph_type * ".txt", graph_name * "_EQW")

        @test g1 == g2
    end

    for n in [4, 16, 36, 64]
        graph_type = "Shastry-Sutherland"
        graph_name = graph_type * "_N$n"

        graph_dat_path = Jun_data_path * "$graph_name.dat"
        edges_raw = readdlm(graph_dat_path, Int)
        g1 = SimpleWeightedGraph(n)
        for e in eachrow(edges_raw)
            SimpleWeightedGraphs.add_edge!(g1, e[1] + 1, e[2] + 1, 1.0)
        end

        g2 = load_weighted_graph(data_path * graph_type * ".txt", graph_name * "_EQW")

        @test g1 == g2
    end
end
