# Numerical aging simulation written in Julia and the Graphs library

using Graphs
using StatsBase
using Printf
using Base.Iterators

function initialize_graph(N, n_sc, n_xl)
    idxs = sample(1:N-1, N-1 - n_sc; replace=false)
    edgs = [Edge(i, i+1) for i in idxs]
    g = SimpleGraphFromIterator(edgs)
    for i in 1:n_xl
        random_xlink!(g)
    end
    return g
end


function random_xlink!(g, coord=nothing)
    max_attempts = 100000
    for i in 1:max_attempts
        v1, v2 = sample(vertices(g), 2, replace=false)
        if degree(g, v1) > max_coordination || degree(g, v2) > max_coordination
            continue
        end
        if add_edge!(g, v1, v2)
            if coord != nothing
                coord[v1] += 1
                coord[v2] += 1
            end
            return
        end
    end
    @error "Maximum attempts exceeded in random_xlink!"
end

function random_scission!(g, coord=degree(g))
    ne(g) == 0 && @error("Graph contains no edges!")
#    v1 = sample(vertices(g), Weights(coord, ne(g)*2))
    v1 = rand(vertices(g))
    while rand(1:max_coordination) > coord[v1]
        v1 = rand(vertices(g))
    end
    v2 = rand(neighbors(g, v1))
    rem_edge!(g, v1, v2)
    coord[v1] -= 1
    coord[v2] -= 1
end


function run!(g, p_xl, niter, nlog=100)
    n_logged = 0
    coord = degree(g)
    scissions = xlinks = 0
    
    println("# iteration gel-% cycle_rank n_bonds")
    
    for i in 1:niter
        if i >= 10^(n_logged/nlog*log(10, niter))
            n_logged += 1
            conn_comps = connected_components(g)
            sizes = map(length, conn_comps)
            gel_comp = conn_comps[argmax(sizes)]
            gel_fraction = length(gel_comp) / nv(g)
            n_gel_edges = sum(degree(g, gel_comp)) / 2
            cycle_rank = n_gel_edges - length(gel_comp) + 1
            n_bonds = ne(g)
            
            @printf("%8d %8f %8d %8d\n", i, 100*gel_fraction, cycle_rank, n_bonds)
        end
        
        if rand() < p_xl
            random_xlink!(g, coord)
        else
            random_scission!(g, coord)
        end
    end
end

const max_coordination = 3

const N = 1000000
Mw = 5600
const initial_scissions = round(Int, N / (Mw/14/2))
const initial_crosslinks = round(Int, initial_scissions*.6)
@printf("# Initial scissions %8d, initial xlinks %8d\n", initial_scissions, initial_crosslinks)

g = initialize_graph(N, initial_scissions, initial_crosslinks)

p_xl = 0.33
niter = N*2
run!(g, p_xl, niter)
