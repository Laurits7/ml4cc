import os
import string
import random
from textwrap import dedent


def generate_run_id(run_id_len=10):
    """ Creates a random alphanumeric string with a length of run_id_len

    Args:
        run_id_len : int
            [default: 10] Length of the alphanumeric string

    Returns:
        random_string : str
            The randomly generated alphanumeric string
    """
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=run_id_len))


def create_tmp_run_dir():
    """ Creates the temporary directory where the slurm job, log and error files are saved for each job

    Args:
        None

    Returns:
        tmp_dir : str
            Path to the temporary directory
    """
    run_id = generate_run_id()
    tmp_dir = os.path.join(os.path.expandvars("$HOME"), "tmp", run_id)
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def prepare_job_file(
        input_file,
        job_idx,
        output_dir,
        run_script
):
    """Writes the job file that will be executed by slurm

    Parameters:
    ----------
    input_file : str
        Path to the file containing the list of .parquet files to be processed
    job_idx : int
        Number of the job.
    output_dir : str
        Directory where the temporary output will be written

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    """
    job_dir = os.path.join(output_dir, "executables")
    os.makedirs(job_dir, exist_ok=True)
    error_dir = os.path.join(output_dir, "error_files")
    os.makedirs(error_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log_files")
    os.makedirs(log_dir, exist_ok=True)
    job_file = os.path.join(job_dir, 'execute' + str(job_idx) + '.sh')
    error_file = os.path.join(error_dir, 'error' + str(job_idx))
    log_file = os.path.join(log_dir, 'output' + str(job_idx))
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            f"""
                #!/bin/bash
                #SBATCH --job-name=ntupelizer
                #SBATCH --ntasks=1
                #SBATCH --partition=short
                #SBATCH --time=02:00:00
                #SBATCH --cpus-per-task=10
                #SBATCH -e {error_file}
                #SBATCH -o {log_file}
                env
                date
                ./run.sh python {run_script} CEPC.preprocessing.slurm.slurm_run=True CEPC.preprocessing.slurm.input_path={input_file}
            """).strip('\n'))
    return job_file


def multipath_slurm_processor(input_path_chunks, job_script):
    tmp_dir = create_tmp_run_dir()
    input_file_paths = create_job_input_list(input_path_chunks, tmp_dir)
    for job_idx, input_file_path in enumerate(input_file_paths):
        prepare_job_file(input_file_path, job_idx, tmp_dir, job_script)
    print(f"Temporary directory created to {tmp_dir}")
    print(f"Run `bash ml4cc/scripts/submit_batchJobs.sh {tmp_dir}/executables/`")


def create_job_input_list(input_path_chunks: list, output_dir: str):
    input_file_paths = []
    for i, in_chunk in enumerate(input_path_chunks):
        input_file_path = os.path.join(output_dir, f"input_paths_{i}.txt")
        with open(input_file_path, 'wt') as outFile:
            for path in in_chunk:
                outFile.write(path + "\n")
        input_file_paths.append(input_file_path)
    return input_file_paths
