function make_sdp(A, C, X_dim, var_dict_constr_pos, var_dict_free_pos)
    model = Model(Mosek.Optimizer)

    @variable(model, X[1:X_dim, 1:X_dim], PSD)

    C_matrix = sparse(C[1], C[2], C[3], X_dim, X_dim)
    @objective(model, Max, dot(C_matrix, X))

    @constraint(model, X[1, 1] == 1.0) # zeroth-order moment should be 1

    [
        @constraint(
            model,
            sum([
                coefficient * X[column_index, row_index] for
                (column_index, row_index, coefficient) in zip(a[1], a[2], a[3])
            ]) == 0.0
        ) for a in values(A)
    ]

    return model
end
