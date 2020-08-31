import numpy as np


def batch_diagonal_averager(elementary_matrix, component_matrix_ref):
    em_rev = elementary_matrix[:, ::-1, :]
    lidx, ridx = -em_rev.shape[1]+1, em_rev.shape[2]
    for i, offset in enumerate(range(lidx, ridx)):
        for t in range(em_rev.shape[0]):
            diag = np.diag(em_rev[t], k=offset)
            component_matrix_ref[t, i] = np.mean(diag)


def elementary_matrix_at_rank(trajectory_matrix,
                              left_singular_vectors,
                              indx):

    U_r = left_singular_vectors[:, indx]
    X_r = np.dot(np.dot(U_r, U_r.T), trajectory_matrix)
    return X_r


def reconstruct_mssa_comps(time_delay_matrix, Va, W, recon_n, n_ts, U):
	components = np.zeros((n_ts, time_delay_matrix.shape[0], 
	                       len(recon_n)))
	for r in recon_n:
		elementary_matrix_r = \
		elementary_matrix_at_rank(time_delay_matrix, Va, r)
		import pdb; pdb.set_trace()
		batch_diagonal_averager(elementary_matrix_r.reshape(n_ts, W, -1),
								components[:, :, r])
	return components.sum(axis=2)


def reconstuct_components(U, Va, W, recon_n, n_ts):
# Borrowed from:
#http://environnement.ens.fr/IMG/file/DavidPDF/SSA_beginners_guide_v9.pdf
    ts_len = U.shape[0]
    RCs = []
    RC = np.zeros((ts_len, n_ts, len(recon_n)))
    for n in recon_n:
        U_rev = time_delay_matrix(U[:,n], W, True)   
        indx=0
        for t in range(n_ts):
            RC[:,t,n] = (U_rev @ Va[n, indx:(indx+W)])/W
            indx+= W
    return RC.sum(axis=2)


def time_delay_matrix(comp_ts, window, reverse=False):
    time_delay_df = pd.DataFrame(comp_ts)
    if reverse:
        time_delay_df = pd.concat([time_delay_df.shift(i) for i in range(window)], 
                                  axis=1)
    else:
        time_delay_df = pd.concat([time_delay_df.shift(-i) for i in range(window)], 
                                  axis=1)
    time_delay_df.fillna(0, inplace=True)
    return time_delay_df.values