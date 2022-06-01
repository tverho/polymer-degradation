# Julia implementation of the KMC degradation simulation

using Graphs
using StatsBase
using Distributions
using Random
using Printf
using Base.Iterators
using Plots


function initialize_chains(N, n_sc)
    idxs = sample(1:N-1, N-1 - n_sc; replace=false)
    edgs = [Edge(i, i+1) for i in idxs]
    return SimpleGraphFromIterator(edgs)
end

function scission!(g, idx)
    edge = first(drop(edges(g), idx-1))
    rem_edge!(g, edge)
    return edge
end

random_scission!(g) = scission!(g, rand(1:ne(g)))

function random_xlink!(g)
    max_coord = 3
    max_attempts = 100000
    for i in 1:max_attempts
        v1, v2 = rand(vertices(g), 2)
        (degree(g, v1) == max_coord || degree(g, v2) == max_coord || v1 == v2) && continue
        if add_edge!(g, v1, v2)
            return v1, v2
        end
    end
    @error "Maximum attempts exceeded in random_xlink!"
end


function KMC_step!(g, T, p_xl=0, max_timestep=100)
    rates = assign_rates(g, T)
    R_sc = sum(rates)
    R_xl = R_sc*p_xl/(1-p_xl)
    R = R_sc + R_xl
    dt = -log(rand())/R
    if dt > max_timestep
        # No reaction
        return max_timestep
    end
    
    if p_xl > 0 && rand() < p_xl
        random_xlink!(g)
    else
        idx = sample(1:ne(g), Weights(rates, R_sc))
        scission!(g, idx)
    end
    return dt
end
    
function assign_rates(g, T)
    A = 1.0e14      # 1/s
    Eact = 248500   # J/mol
    R = 8.314       # J/mol/K
    
    rate = A*exp(-Eact/(R*T))
    rates = fill(rate, ne(g))
    return rates
end

function largest_volatile(T)
    w0 = 67.328
    w1 = 1191.8
    w2 = 0.90918
    w3 = 20.941
    return w3/((w1-w0)/(T-w0)-1)^(1/w2)
end

function run!(g, Tstart, Tend, beta, t_log; p_xl=0)
    i = 0
    t = 0
    T = Tstart
    lastlog = -Inf

    println("# iteration  time  temperature  solid-%  gel-%  cycle_rank  n_bonds")
    
    while T < Tend
        dt = KMC_step!(g, T, p_xl)
        
        if t - lastlog > t_log
            conn_comps = connected_components(g)
            sizes = map(length, conn_comps)
            gel_comp = conn_comps[argmax(sizes)]
            gel_fraction = length(gel_comp) / nv(g)
            n_gel_edges = sum(degree(g, gel_comp)) / 2
            cycle_rank = n_gel_edges - length(gel_comp) + 1
            
            volatile_limit = largest_volatile(T)
            m_volatile = sum(filter(s -> s<=volatile_limit, sizes))
            solid_fraction = 1 - m_volatile/nv(g)
            n_bonds = ne(g)    
            @printf("%8d %8f %8f %8f %8f %8d %8d\n", i, t, T-273.15, 100*solid_fraction, 100*gel_fraction, cycle_rank, n_bonds)
            lastlog = t
        end
        
        t += dt
        T += beta*dt
        i += 1
    end
end


const N = 100000
const mean_length = 4000/14/2
const initial_scissions = round(Int, N/mean_length)
const initial_crosslinks = 0.4/65 * N

const Tstart = 100 + 273.15
const Tend = 550 + 273.15
const beta = 10/60

g = initialize_chains(N, initial_scissions)

conn_comps = connected_components(g)
weights = map(length, conn_comps) * 14
Mn = mean(weights)
Mw = sum(weights.^2)/sum(weights)
println("# Mn=$Mn  Mw=$Mw  dispersity=$(Mw/Mn)  before cross-linking")


for i in 1:initial_crosslinks
    random_xlink!(g)
end


run!(g, Tstart, Tend, beta, 1.0)
