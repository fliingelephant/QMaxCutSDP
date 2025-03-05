function load_weighted_graph(gpath::String, gname::String)
    open(gpath, "r") do io
        return loadgraph(io, gname, SWGFormat())
    end
end

function _save_weighted_graph(gpath::String, g::SimpleWeightedGraph, gname::String)
    open(gpath, "a+") do io
        return savegraph(io, g, gname, SWGFormat())
    end
end
