import itertools as it
import functools as ft
import numpy as np
import scipy as sp
from typing import List, Tuple
import time
import mosek
from   mosek.fusion import *
import sys
from pathlib import Path

def get_data_weighted(n, edges):
    r"""
    Get all the data necessary for mosek fusion modeling in the primal formulation:
            max  sum_{(i,j)\in E} w_{ij} * h_{ij}
            s.t. M(h_{ij}) >= 0,
    where M is the moment matrix, and '>=' stands for positive semidefiniteness. 

    Matrix indexing is column based. 
    
    Edges are weighted, i.e., edges = [(i_1, j_1, w_{i_1j_1}),...,(i_s, j_s, w_{i_sj_s})].

    """

    # Get all the h tuples (i,j), i < j, and (0,0)
    htuples = [(0,0)] + list(it.combinations(range(n),2))
    n_htuples = len(htuples)
    # Dictionary storing the positions of the constraint variables, i.e., all the h_{ij}'s. 
    var_dict_constr_pos = {h: (i, 0) for i, h in enumerate(htuples)}
    # Dictionary storing the positions of the free variables, i.e., all the h_{ij}h_{kl}, 
    # where there's no overlap between {i,j} and {k, l}.
    var_dict_free_pos = {}
    # A stores all sparse matrices that specify the equality constraints in moment matrix M such that Tr(M A_i) = 0
    A = {}
    for i in range(n_htuples):
        for j in range(i, n_htuples):
            a, b = htuples[i], htuples[j]
            if i == 0 and j >= 1:
                A[(a,b)] = [[], [], []]
                A[(a,b)][0] = [j, j]
                A[(a,b)][1] = [i, j]
                A[(a,b)][2] = [1, -1]
            elif i > 0 and j > i: 
                a_set = set(a)
                b_set = set(b)
                c = a_set.intersection(b_set)
                u = a_set.union(b_set)
                if len(c) == 0:
                    var_dict_free_pos[(a,b)] = (j,i)
                elif len(c) == 1:
                    # get the single element from set c
                    (e,) = c
                    u.remove(e)
                    pos_a = var_dict_constr_pos[a]
                    pos_b = var_dict_constr_pos[b]
                    pos_c = var_dict_constr_pos[tuple(sorted(tuple(u)))]
                    A[(a,b)] = [[], [], []]
                    A[(a,b)][0] = [j, pos_a[0], pos_b[0], pos_c[0]]
                    A[(a,b)][1].extend([i, pos_a[1], pos_b[1], pos_c[1]])
                    A[(a,b)][2].extend([-1, 1/4, 1/4, -1/4])

    # Obtain matrix C in sparse format
    C = [[], [], []]
    for i, j, w in edges:
        pos = var_dict_constr_pos[(i,j)]
        C[0].append(pos[0])
        C[1].append(pos[1])
        C[2].append(w)

    # Dimension of moment matrix M
    X_dim = int(n * (n-1) / 2 + 1)

    return A, C, X_dim, var_dict_constr_pos, var_dict_free_pos

def solve_main_weighted(n, edges):
    """Main functioin for level-1 projector SDP for QMAXCUT. 

    Edges are weighted, i.e., edges = [(i_1, j_1, w_{i_1j_1}),...,(i_s, j_s, w_{i_sj_s})].
    
    """

    A, C, X_dim, var_dict_constr_pos, var_dict_free_pos = get_data_weighted(n, edges)

    with Model("qmc") as M:
        
        # Setting up the variables
        X = M.variable("X", Domain.inPSDCone(X_dim))

        # Objective
        C = Matrix.sparse(X_dim, X_dim, C[0], C[1], C[2])
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(C, X))

        # Constraints
        B = [ Matrix.sparse(X_dim, X_dim, a[0], a[1], a[2]) for a in A.values() ]
        M.constraint(X.index([0,0]), Domain.equalsTo(1.0))
        [ M.constraint(Expr.dot(b, X), Domain.equalsTo(0.0)) for b in B ]

        # Solve
        M.setLogHandler(sys.stdout)
        M.solve()

        # Get the objective value, moments, and the moment matrix (=S)
        obj_val = M.primalObjValue()
        X_sol = X.level()
        # moment matrix
        MM = np.reshape(X_sol, (X_dim,X_dim))
        # Correlations, first row of the moment matrix
        # correlations = {var:  MM[pos[0], pos[1]] for var, pos in var_dict_constr_pos.items()}
        # free moments, i.e., <h_ij h_kl> where no overlap between {i,j} and {k,l}
        # moments_free = {var: MM[pos[0], pos[1]] for var, pos in var_dict_free_pos.items()}
        # All the unique moments
        # moments = correlations | moments_free

    return obj_val, MM


if __name__=='__main__':


    # Example: ring-n with uniform weights 1.
    # n = 4
    # edges = [(i, i+1, 1) for i in range(n-1)] + [(0,n-1,1)]
    # graph_name = f'ring-{n}'
    # print(edges)

    # dir_result = './ring/'

    # Jun's graphs
    n = 32
    graph_name = f'Majumdar-Ghosh_N{n}'
    # graph_name = f'Shastry-Sutherland_N{n}'
    graph_path = f'./graphdata/{graph_name}.dat'
    edges = np.loadtxt(graph_path, dtype=int).tolist()
    edges = [tuple(sorted(e[0:2]) + [1]) for e in edges]
    dir_result = f'./{graph_name}/'
    # Make file path if not existed
    Path(dir_result).mkdir(parents=True, exist_ok=True)
    time_start = time.time()
    E_proj, moment_matrix = solve_main_weighted(n, edges)
    time_end_proj = time.time()
    t_proj = np.round(time_end_proj - time_start, 2)