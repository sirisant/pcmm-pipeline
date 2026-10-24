from PCMM.phase_coherence_kmeans import diametrical_clustering
import os
import numpy as np

def run_modeling(data_dir, output_dir, K, num_repl, max_iter, tol, init, atlas_name):
    print(f"Running modeling with K={K}, num_repl={num_repl}, max_iter={max_iter}, tol={tol}, init={init}")
    os.makedirs(output_dir, exist_ok=True)
    Z_path = os.path.join(data_dir, f'Z_{atlas_name}.npy')
    Z = np.load(Z_path)

    C, X_part, obj = diametrical_clustering(X=Z,K=K,max_iter=max_iter,num_repl=num_repl,tol=tol,init=init)

    np.savetxt(os.path.join(output_dir, f"C_K{K}.txt"), C)
    np.savetxt(os.path.join(output_dir, f"X_part_K{K}.txt"), X_part)
    np.savetxt(os.path.join(output_dir, f"obj_K{K}.txt"), obj)

    print(f"Saved complex diametrical clustering outputs to {output_dir}")