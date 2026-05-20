import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from classy import Class
from itertools import product
from tqdm import tqdm
import sys
import gc
import traceback

sys.path.append('/home/pedrorozin/paper_tesis2025/source/')
from main_functions import common_settings, compute_delta_m, k_horizon, read_adhoc_txt, deriv_tau_to_a, get_sigma8


def main():
    """
    Main function to build the parameter grid. This script extracts the full vectors 
    for initial conditions.
    
    CLASS calculates perturbations for all k modes, so 'k' is not a parameter we sweep; 
    instead, we filter the resulting dataframe for k >= k_horizon.
    
    Parameters to sweep:
    - Omega_m (Total matter density related)
    - h (Dimensionless Hubble parameter)

    Pipeline:
    1. Set up parameter ranges and create output directories.
    2. Iterate through combinations of omega_m and h using a Cartesian product.
    3. Initialize the CLASS universe given a set of parameters with `common_settings`.
    4. Call `get_perturbations()` to compute perturbations for that universe.
    5. Read the adhoc text file with `read_adhoc_txt` to obtain perturbations and their derivatives.
    6. Filter the dataframe by `a_ini`: keep values where 'a' >= a_ini and drop duplicates.
    7. Calculate `k_horizon()` to get the exact horizon scale k (in h/Mpc).
    8. Filter the DataFrame to retain only modes that are inside the horizon (k >= k_horizon).
    9. Calculate sigma8 for the current cosmology using `get_sigma8()`.
    10. Iterate over the remaining unique k modes:
        - Apply `deriv_tau_to_a()` to compute derivatives with respect to 'a'.
        - Extract initial conditions (at the minimum available 'a' for each k).
        - Compute total matter perturbations and build a dictionary with the results.
    11. Incrementally append the results to a CSV file to optimize memory usage.
    12. Clean up CLASS C-structures to prevent memory leaks.
    13. Delete the temporary adhoc file to safely generate a new one in the next iteration.
    14. Generate a summary statistics text file at the end of the run.
    """

    # Folder where the grid will be saved. Check if it exists, if not create it.
    path_folder = '/home/pedrorozin/paper_tesis2025/outputs/grids/'
    n = 'grid_z_approx_100'

    error_log = f'{path_folder}/{n}/errors_log_{n}.txt' # Where errors will be logged

    if os.path.exists(path_folder + n):
        raise FileExistsError(f"The directory {path_folder}/{n} already exists.")
    else:
        os.makedirs(path_folder + n)
    #==========================

    # # Range of values for each parameter
    omega_m_values = np.arange(0.153, 0.453, 0.005)
    h_values = np.arange(0.643, 0.763, 0.005)

    # Range of values for each parameter for validation
    # omega_m_values = np.arange(0.163, 0.443, 0.005)
    # h_values = np.arange(0.653, 0.743, 0.003)
    
    # omega_m_values = np.arange(0.30, 0.32, 0.01)
    # h_values = np.arange(0.65, 0.67, 0.01)
    # k_values = np.arange(0.02, 0.22, 0.02)
    
    results = []
    
    #choose the initial momento for the integrations. This is crucial for the results, as it determines the initial conditions for the perturbations.
    #the moment can be chosen by a_ini or z_ini, but we will use a_ini for the calculations. 
    
    z_ini= 100
    # a_ini = 0.03 # z \approx 33
    a_ini = 1/(1+z_ini)
    
    # Output file to save results incrementally
    output_file = f'{path_folder}/{n}/grilla_results_{n}.csv'
    file_exists = False  # To check if it already has a header
    
    for omega_m, h in tqdm(product(omega_m_values, h_values)):
        
        try:
            # 3. Set a Universe given a set of parameters with `common_settings`.
            M = common_settings(k=0.1, omega_m=omega_m, h=h) # This is actually CLASS Omega_m, not omega_m

            # 4. get_perturbations() to obtain the perturbations of that universe.
            # This executes CLASS.compute() and returns perturbations in the adhoc file.
            _perturbations = M.get_perturbations() # Dummy variable. Only serves to execute CLASS compute().
            
            # 5. Read the text file with `read_adhoc_txt` to obtain the perturbations and their derivatives.
            df = read_adhoc_txt(file_path='/home/pedrorozin/scripts/class_pedro/delta_prime_cdm.txt')
            
            # Filter DF with a_ini: keep the values of a >= a_ini. 
            df = df[df['a'] >= a_ini]
            # Sort by 'a' and drop duplicates
            df = df.drop_duplicates(subset=['a'], keep='first').sort_values('a')

            # 7. Calculate k_horizon() to get the horizon scale k.
            a_ini_actual = df['a'].min()  # The minimum 'a' after initial filters
            k_hor = k_horizon(a_ini=a_ini_actual, omega_m=omega_m, h= h) # c in km/s, k_hor in h/Mpc
            df['k h'] = df['k'] / h # k to h/Mpc
            
            # Drop all kh > 0.25 (Currently commented out)
            # df = df[df['k h'] <= 0.25]
            
            # 8. Filter the DataFrame to get only perturbations with k >= k_horizon.
            df_filtered = df[df['k h'] >= k_hor].copy()

            uniques_ks = df_filtered['k'].unique()
            
            sigma8 = get_sigma8(M)
            
            # Extract parameters
            omega_b = M.Omega_b()
            _omega_m = M.Omega_m() # Should be the same as the iteration value
            omega_cdm = omega_m - omega_b

            for _k in uniques_ks:
                # Filter by specific k
                df_k = df_filtered[df_filtered['k'] == _k].copy()
                
                # Apply derivatives only to this k
                df_k = deriv_tau_to_a(df_k, column_name='delta_dot_cdm')
                df_k = deriv_tau_to_a(df_k, column_name='delta_dot_b')
                
                # Get the index of the minimum 'a' for this k
                min_a_idx = df_k['a'].idxmin()
                
                # Extract values for this specific k
                delta_cdm = np.float128(df_k.loc[min_a_idx, 'delta_cdm'])
                delta_b = np.float128(df_k.loc[min_a_idx, 'delta_b'])
                delta_m = compute_delta_m(delta_cdm, delta_b, omega_cdm, omega_b)
                delta_prime_cdm = np.float128(df_k.loc[min_a_idx, 'delta_prime_cdm'])
                delta_prime_b = np.float128(df_k.loc[min_a_idx, 'delta_prime_b'])
                delta_prime_m = compute_delta_m(delta_prime_cdm, delta_prime_b, omega_cdm, omega_b)

                result_dict = {
                    'a': df_k.loc[min_a_idx, 'a'],  
                    'k': df_k.loc[min_a_idx, 'k'],  # Original k
                    'k h': df_k.loc[min_a_idx, 'k h'],
                    'Omega_cdm': omega_cdm,
                    'Omega_b': omega_b,
                    'Omega_m': omega_m,
                    'A_s': 2.e-9,  # Fixed A_s
                    'h': h,
                    'k_horizon': k_hor,
                    'sigma8': sigma8,
                    'delta_cdm': delta_cdm,  
                    'delta_b': delta_b,      
                    'delta_m': delta_m,
                    'delta_prime_cdm': delta_prime_cdm,  
                    'delta_prime_b': delta_prime_b,
                    'delta_prime_m': delta_prime_m
                }
                results.append(result_dict)
            
            # 11. Save results of this iteration immediately
            if results:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(output_file, mode='a', header=not file_exists, index=False)
                file_exists = True
                results = []  # Clear list to free memory
            
        except Exception as e:
            err_msg = f"Error for omega_m={omega_m}, h={h}: {str(e)}\n"
            print(f'\n{err_msg}')
            with open(error_log, 'a') as f:
                f.write(err_msg)
                f.write(traceback.format_exc() + "\n")
                
        finally:  # Clean up everything
            try:
                M.struct_cleanup()
                del M
            except:
                pass

    # 13. Delete the adhoc file to generate a new one in the next iteration.
    if os.path.exists('/home/pedrorozin/scripts/class_pedro/delta_prime_cdm.txt'):
        os.remove('/home/pedrorozin/scripts/class_pedro/delta_prime_cdm.txt')
    gc.collect()
        
    # 14. Read final results to generate statistics
    if os.path.exists(output_file):
        df_results = pd.read_csv(output_file)
    else:
        df_results = pd.DataFrame()
        print("Warning: No results were generated.")
    
    # Create info_grilla.txt
    with open(f'{path_folder}/{n}/info_grilla_{n}.txt', 'w') as f:
        f.write('Parameter grid:\n')
        f.write(f'Omega_m: min {omega_m_values.min()}, max {omega_m_values.max()}, length {len(omega_m_values)}\n')
        f.write(f'h: min {h_values.min()}, max {h_values.max()}, length {len(h_values)}\n')
        f.write('A_s fixed at 2.e-9\n')
        if not df_results.empty:
            f.write(f'Total number of points in the grid: {len(df_results)}\n')
            f.write(f'min a_ini: {df_results["a"].min()}, max a_ini: {df_results["a"].max()}\n')
            f.write(f'k_horizon range: min {df_results["k_horizon"].min()} h/Mpc, max {df_results["k_horizon"].max()} h/Mpc\n')
        else:
            f.write('No results were generated (check error log).\n')

if __name__ == "__main__":
    main()