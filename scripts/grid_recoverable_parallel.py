import gc
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

# Keep CPU workers single-threaded so parallel processes do not oversubscribe the machine.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('/home/pedrorozin/paper_tesis2025/source/')
from main_functions import common_settings, compute_delta_m, get_sigma8, k_horizon, read_adhoc_txt

RESUME_PRECISION = 12
RESULT_COLUMNS = [
    'a',
    'k',
    'k h',
    'Omega_cdm',
    'Omega_b',
    'Omega_m',
    'A_s',
    'h',
    'k_horizon',
    'sigma8',
    'delta_cdm',
    'delta_b',
    'delta_m',
    'delta_prime_cdm',
    'delta_prime_b',
    'delta_prime_m',
]


def _pair_key(omega_m, h, precision=RESUME_PRECISION):
    return f'{omega_m:.{precision}f}|{h:.{precision}f}'


def _chunk_pairs(pairs, block_size):
    for start in range(0, len(pairs), block_size):
        yield pairs[start:start + block_size]


def _empty_summary():
    return {
        'rows': 0,
        'a_min': None,
        'a_max': None,
        'k_horizon_min': None,
        'k_horizon_max': None,
    }


def _update_summary(summary, df_chunk):
    if df_chunk.empty:
        return summary

    summary['rows'] += len(df_chunk)
    a_min = float(df_chunk['a'].min())
    a_max = float(df_chunk['a'].max())
    k_min = float(df_chunk['k_horizon'].min())
    k_max = float(df_chunk['k_horizon'].max())

    summary['a_min'] = a_min if summary['a_min'] is None else min(summary['a_min'], a_min)
    summary['a_max'] = a_max if summary['a_max'] is None else max(summary['a_max'], a_max)
    summary['k_horizon_min'] = k_min if summary['k_horizon_min'] is None else min(summary['k_horizon_min'], k_min)
    summary['k_horizon_max'] = k_max if summary['k_horizon_max'] is None else max(summary['k_horizon_max'], k_max)
    return summary


def _merge_summary_dict(summary, block_summary):
    if block_summary['rows'] <= 0:
        return summary

    summary['rows'] += block_summary['rows']

    if block_summary['a_min'] is not None:
        summary['a_min'] = block_summary['a_min'] if summary['a_min'] is None else min(summary['a_min'], block_summary['a_min'])
    if block_summary['a_max'] is not None:
        summary['a_max'] = block_summary['a_max'] if summary['a_max'] is None else max(summary['a_max'], block_summary['a_max'])
    if block_summary['k_horizon_min'] is not None:
        summary['k_horizon_min'] = block_summary['k_horizon_min'] if summary['k_horizon_min'] is None else min(summary['k_horizon_min'], block_summary['k_horizon_min'])
    if block_summary['k_horizon_max'] is not None:
        summary['k_horizon_max'] = block_summary['k_horizon_max'] if summary['k_horizon_max'] is None else max(summary['k_horizon_max'], block_summary['k_horizon_max'])

    return summary


def _save_checkpoint(checkpoint_file, completed_pairs):
    payload = {
        'completed_pairs': sorted(completed_pairs),
        'updated_at': time.time(),
    }
    tmp_file = checkpoint_file.with_name(checkpoint_file.name + '.tmp')
    with open(tmp_file, 'w') as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_file, checkpoint_file)


def _load_completed_pairs_from_checkpoint(checkpoint_file):
    if not checkpoint_file.exists():
        return set()

    with open(checkpoint_file) as handle:
        payload = json.load(handle)

    return set(payload.get('completed_pairs', []))


def _load_completed_pairs_from_csv(output_file, resume_precision=RESUME_PRECISION):
    if not output_file.exists() or output_file.stat().st_size == 0:
        return set()

    try:
        completed_df = pd.read_csv(output_file, usecols=['Omega_m', 'h'])
    except Exception:
        return set()

    return {
        _pair_key(omega_m, h, precision=resume_precision)
        for omega_m, h in completed_df.drop_duplicates().itertuples(index=False, name=None)
    }


def _bootstrap_summary_from_csv(output_file):
    summary = _empty_summary()
    if not output_file.exists() or output_file.stat().st_size == 0:
        return summary

    try:
        df_results = pd.read_csv(output_file, usecols=['a', 'k_horizon'])
    except Exception:
        return summary

    if df_results.empty:
        return summary

    summary['rows'] = len(df_results)
    summary['a_min'] = float(df_results['a'].min())
    summary['a_max'] = float(df_results['a'].max())
    summary['k_horizon_min'] = float(df_results['k_horizon'].min())
    summary['k_horizon_max'] = float(df_results['k_horizon'].max())
    return summary


def _remove_file_if_exists(file_path):
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


def _build_pair_results(df_filtered, omega_m, h, omega_b, omega_cdm, sigma8, k_hor):
    df_filtered = df_filtered.copy()
    df_filtered['delta_prime_cdm'] = df_filtered['delta_dot_cdm'] / (df_filtered['H'] * df_filtered['a'])
    df_filtered['delta_prime_b'] = df_filtered['delta_dot_b'] / (df_filtered['H'] * df_filtered['a'])

    selected = df_filtered.loc[df_filtered.groupby('k')['a'].idxmin()].copy()
    selected['delta_m'] = compute_delta_m(
        selected['delta_cdm'].to_numpy(),
        selected['delta_b'].to_numpy(),
        omega_cdm,
        omega_b,
    )
    selected['delta_prime_m'] = compute_delta_m(
        selected['delta_prime_cdm'].to_numpy(),
        selected['delta_prime_b'].to_numpy(),
        omega_cdm,
        omega_b,
    )

    selected['Omega_cdm'] = omega_cdm
    selected['Omega_b'] = omega_b
    selected['Omega_m'] = omega_m
    selected['A_s'] = 2.e-9
    selected['h'] = h
    selected['k_horizon'] = k_hor
    selected['sigma8'] = sigma8

    return selected[RESULT_COLUMNS]


def _process_block(block_payload):
    block_id, pair_block, output_dir_str, a_ini, resume_precision = block_payload
    output_dir = Path(output_dir_str)
    block_output_dir = output_dir / '_blocks'
    work_dir = output_dir / '_workdirs' / f'block_{block_id:05d}'
    block_output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    prev_cwd = Path.cwd()
    block_rows = []
    completed_pairs = []
    errors = []
    summary = _empty_summary()
    block_csv = block_output_dir / f'block_{block_id:05d}.csv'
    adhoc_file = work_dir / 'delta_prime_cdm.txt'

    try:
        os.chdir(work_dir)

        for omega_m, h in pair_block:
            pair_key = _pair_key(omega_m, h, precision=resume_precision)
            M = None

            try:
                iter_start = time.time()
                M = common_settings(k=0.1, omega_m=omega_m, h=h)
                _perturbations = M.get_perturbations()

                df = read_adhoc_txt(file_path=str(adhoc_file))
                df = df.loc[df['a'] >= a_ini].drop_duplicates(subset=['a'], keep='first').sort_values('a')

                if df.empty:
                    completed_pairs.append(pair_key)
                    _remove_file_if_exists(adhoc_file)
                    continue

                a_ini_actual = df['a'].min()
                k_hor = k_horizon(a_ini=a_ini_actual, omega_m=omega_m, c=3e5, h=h)
                df['k h'] = df['k'] / h
                k_mask = (df['k h'] >= (k_hor / 800)) & (df['k h'] <= 0.5)
                df_filtered = df.loc[k_mask].copy()
                

                if df_filtered.empty:
                    completed_pairs.append(pair_key)
                    _remove_file_if_exists(adhoc_file)
                    continue

                sigma8 = get_sigma8(M)
                omega_b = M.Omega_b()
                omega_cdm = omega_m - omega_b

                pair_results = _build_pair_results(
                    df_filtered=df_filtered,
                    omega_m=omega_m,
                    h=h,
                    omega_b=omega_b,
                    omega_cdm=omega_cdm,
                    sigma8=sigma8,
                    k_hor=k_hor,
                )

                if not pair_results.empty:
                    block_rows.append(pair_results)
                    summary = _update_summary(summary, pair_results)

                completed_pairs.append(pair_key)
                _remove_file_if_exists(adhoc_file)

            except Exception as pair_error:
                errors.append({
                    'omega_m': omega_m,
                    'h': h,
                    'error': str(pair_error),
                    'traceback': traceback.format_exc(),
                })
            finally:
                try:
                    if M is not None:
                        M.struct_cleanup()
                        del M
                except Exception:
                    pass

        if block_rows:
            block_df = pd.concat(block_rows, ignore_index=True)
            block_df.to_csv(block_csv, index=False)
        else:
            block_csv = None

        return {
            'block_id': block_id,
            'block_csv': str(block_csv) if block_csv is not None else None,
            'completed_pairs': completed_pairs,
            'summary': summary,
            'errors': errors,
        }

    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(work_dir, ignore_errors=True)


def _append_block_results(output_file, block_csv):
    if block_csv is None:
        return output_file.exists() and output_file.stat().st_size > 0

    block_path = Path(block_csv)
    if not block_path.exists() or block_path.stat().st_size == 0:
        return output_file.exists() and output_file.stat().st_size > 0

    block_df = pd.read_csv(block_path)
    if block_df.empty:
        return output_file.exists() and output_file.stat().st_size > 0

    file_exists = output_file.exists() and output_file.stat().st_size > 0
    block_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    return True


def _write_info_file(output_dir, n, omega_m_values, h_values, summary, block_size, parallel_workers):
    with open(output_dir / f'info_grilla_{n}.txt', 'w') as handle:
        handle.write('Parameter grid:\n')
        handle.write(f'Omega_m: min {omega_m_values.min()}, max {omega_m_values.max()}, length {len(omega_m_values)}\n')
        handle.write(f'h: min {h_values.min()}, max {h_values.max()}, length {len(h_values)}\n')
        handle.write('A_s fixed at 2.e-9\n')
        handle.write(f'Block size: {block_size}\n')
        handle.write(f'Parallel workers: {parallel_workers}\n')
        if summary['rows'] > 0:
            handle.write(f'Total number of points in the grid: {summary["rows"]}\n')
            handle.write(f'min a_ini: {summary["a_min"]}, max a_ini: {summary["a_max"]}\n')
            handle.write(f'k_horizon range: min {summary["k_horizon_min"]} h/Mpc, max {summary["k_horizon_max"]} h/Mpc\n')
        else:
            handle.write('No results were generated (check error log).\n')


def _detect_parallel_defaults():
    detected_cores = os.cpu_count() or 1  # Number of logical CPU cores reported by the OS.
    reserved_cores = 2 if detected_cores >= 6 else 1  # Leave a small margin for the OS and background I/O.
    usable_cores = max(1, detected_cores - reserved_cores)  # Cores we are comfortable using for CLASS workers.
    default_workers = max(1, min(8, max(1, usable_cores // 2)))  # Split usable cores across a moderate number of workers.
    default_block_size = max(4, min(16, max(4, usable_cores // default_workers)))  # Keep each worker busy with a non-trivial chunk.
    return detected_cores, usable_cores, default_workers, default_block_size


def main():
    """
    Recoverable grid runner optimized for long runs.

    Key improvements:
    - resumes from a checkpoint and the existing CSV;
    - processes the parameter grid in blocks;
    - uses process-based parallelism across blocks;
    - keeps CLASS isolated in per-worker temporary directories;
    - vectorizes the per-k postprocessing inside each pair.
    """

    path_folder = '/home/pedrorozin/paper_tesis2025/outputs/grids/'
    n = 'grid_z_approx_32_training_data_v3'

    output_dir = Path(path_folder) / n
    output_dir.mkdir(parents=True, exist_ok=True)

    error_log = output_dir / f'errors_log_{n}.txt'
    output_file = output_dir / f'grilla_results_{n}.csv'
    checkpoint_file = output_dir / f'checkpoint_{n}.json'
    resume_precision = RESUME_PRECISION

    # training grid.
    omega_m_values = np.arange(0.084, 0.462, 0.002)
    h_values = np.arange(0.62, 0.772, 0.002)

    #validation grid
    # omega_m_values = np.arange(0.154, 0.444, 0.0012)
    # h_values = np.arange(0.634, 0.754, 0.0012)

    a_ini = 0.0295  # z \approx 33

    detected_cores, usable_cores, default_workers, default_block_size = _detect_parallel_defaults()  # Inspect CPU capacity once and derive safe defaults.
    block_size = max(1, int(os.environ.get('GRID_BLOCK_SIZE', str(default_block_size))))  # Allow manual override, otherwise use the core-based default.
    parallel_workers = max(1, int(os.environ.get('GRID_PARALLEL_WORKERS', str(default_workers))))  # Allow manual override, otherwise use the core-based default.
    parallel_workers = min(parallel_workers, usable_cores)  # Never spawn more workers than the usable CPU cores we detected.

    completed_pairs = _load_completed_pairs_from_checkpoint(checkpoint_file)
    completed_pairs |= _load_completed_pairs_from_csv(output_file, resume_precision=resume_precision)
    summary = _bootstrap_summary_from_csv(output_file)

    all_pairs = [(omega_m, h) for omega_m, h in product(omega_m_values, h_values)]
    remaining_pairs = [pair for pair in all_pairs if _pair_key(pair[0], pair[1], precision=resume_precision) not in completed_pairs]
    blocks = list(_chunk_pairs(remaining_pairs, block_size))

    if completed_pairs:
        print(f'Resuming from {len(completed_pairs)} completed (Omega_m, h) pairs.', flush=True)

    if not blocks:
        _write_info_file(output_dir, n, omega_m_values, h_values, summary, block_size, parallel_workers)
        return

    print(
        f'Detected {detected_cores} logical core(s); using {usable_cores} usable core(s), '
        f'block size {block_size}, and {parallel_workers} worker(s).',
        flush=True,
    )

    block_payloads = [
        (block_id, block, str(output_dir), a_ini, resume_precision)
        for block_id, block in enumerate(blocks, start=1)
    ]

    use_parallel = parallel_workers > 1 and len(block_payloads) > 1

    if use_parallel:
        mp_context = __import__('multiprocessing').get_context('spawn')
        executor = ProcessPoolExecutor(max_workers=parallel_workers, mp_context=mp_context)
        futures = [executor.submit(_process_block, payload) for payload in block_payloads]

        try:
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                completed_pairs.update(result['completed_pairs'])
                summary = _merge_summary_dict(summary, result['summary'])

                if result['block_csv'] is not None:
                    _append_block_results(output_file, result['block_csv'])
                    Path(result['block_csv']).unlink(missing_ok=True)

                for error_item in result['errors']:
                    err_msg = (
                        f"Error in block {result['block_id']} for omega_m={error_item['omega_m']}, "
                        f"h={error_item['h']}: {error_item['error']}\n"
                    )
                    print(f'\n{err_msg}')
                    with open(error_log, 'a') as handle:
                        handle.write(err_msg)
                        handle.write(error_item['traceback'] + '\n')

                _save_checkpoint(checkpoint_file, completed_pairs)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
    else:
        for payload in tqdm(block_payloads, total=len(block_payloads)):
            result = _process_block(payload)
            completed_pairs.update(result['completed_pairs'])
            summary = _merge_summary_dict(summary, result['summary'])

            if result['block_csv'] is not None:
                _append_block_results(output_file, result['block_csv'])
                Path(result['block_csv']).unlink(missing_ok=True)

            for error_item in result['errors']:
                err_msg = (
                    f"Error in block {result['block_id']} for omega_m={error_item['omega_m']}, "
                    f"h={error_item['h']}: {error_item['error']}\n"
                )
                print(f'\n{err_msg}')
                with open(error_log, 'a') as handle:
                    handle.write(err_msg)
                    handle.write(error_item['traceback'] + '\n')

            _save_checkpoint(checkpoint_file, completed_pairs)

    _save_checkpoint(checkpoint_file, completed_pairs)

    if output_file.exists() and output_file.stat().st_size > 0 and summary['rows'] == 0:
        summary = _bootstrap_summary_from_csv(output_file)

    _write_info_file(output_dir, n, omega_m_values, h_values, summary, block_size, parallel_workers)

    gc.collect()


if __name__ == '__main__':
    main()
