import concurrent.futures
import glob
import multiprocessing as mp
import os
import re
import subprocess
import time

import generate_eff
import matplotlib.pyplot as plt
import pandas as pd
import settings
import wandb
from model import *


def extract_r_values_from_pdb(pdb_file_path: str) -> tuple[float, float]:

    r_work, r_free = None, None
    try:
        with open(pdb_file_path, "r") as f:
            for line in f:
                if "REMARK   3   R VALUE (WORKING SET)" in line:
                    r_work = float(line.strip().split(":")[1])
                if "REMARK   3   R VALUE (FREE   SET)" in line:
                    r_free = float(line.strip().split(":")[1])
    except Exception as e:
        print(f"Failed to find r-values: {e}")
    return r_work, r_free


def extract_r_values_from_log(log_file_path: str) -> tuple:

    pattern1 = re.compile(r"Start R-work")
    pattern2 = re.compile(r"Final R-work")

    try:
        with open(log_file_path, "r") as f:
            lines = f.readlines()
        print("opened log file")

        match = re.search(r"epoch=(\d+)", log_file_path)
        if match:
            epoch = match.group(1)
            epoch_start = epoch_final = float(epoch)
        else:
            epoch_start = epoch_final = 0
        print("match lines")
        matched_lines_start = [line.strip() for line in lines if pattern1.search(line)]
        matched_lines_final = [line.strip() for line in lines if pattern2.search(line)]

        # Initialize all values as None

        rwork_start = rwork_final = rfree_start = rfree_final = None

        if matched_lines_start:
            print("findall")
            numbers = re.findall(r"\d+\.\d+", matched_lines_start[0])
            if len(numbers) >= 2:
                print("access rs")
                rwork_start = float(numbers[0])
                rfree_start = float(numbers[1])
            else:
                print(
                    f"Start R-work line found but not enough numbers: {matched_lines_start[0]}"
                )
        else:
            print("No Start R-work line found in log.")

        if matched_lines_final:
            numbers = re.findall(r"\d+\.\d+", matched_lines_final[0])
            if len(numbers) >= 2:
                rwork_final = float(numbers[0])
                rfree_final = float(numbers[1])
            else:
                print(
                    f"Final R-work line found but not enough numbers: {matched_lines_final[0]}"
                )
        else:
            print("No Final R-work line found in log.")
        print("finished r-val extraction")

        return (
            epoch_start,
            epoch_final,
            [rwork_start, rwork_final],
            [rfree_start, rfree_final],
        )
    except Exception as e:
        print(f"Could not open log file: {e}")
        return None, None, [None, None], [None, None]


def run_phenix(repo_dir: str, checkpoint_name: str, mtz_output_path: str):
    phenix_env = (
        "/n/hekstra_lab/garden_backup/phenix-1.21/phenix-1.21.1-5286/phenix_env.sh"
    )
    phenix_output_dir = os.path.join(repo_dir, "phenix_output")
    os.makedirs(phenix_output_dir, exist_ok=True)
    path_to_pdb_model = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/9b7c.pdb"

    # mtz_output_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/merged.mtz"  # for the reference peaks

    phenix_refine_eff_file_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/phenix.eff"

    r_free_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/rfree.mtz"
    pdb_model_path = "/n/holylabs/LABS/hekstra_lab/Users/fgiehr/jobs/anomalous_peaks_files/pdb_model/9b7c.pdb"

    specific_phenix_eff_file_path = os.path.join(
        repo_dir, f"phenix_{checkpoint_name}.eff"
    )
    phenix_out_name = f"refine_{checkpoint_name}"

    subprocess.run(
        [
            "python",
            "/n/holylabs/hekstra_lab/Users/fgiehr/factory/src/factory/generate_eff.py",
            "--sf-mtz",
            mtz_output_path,
            "--rfree-mtz",
            r_free_path,
            "--out",
            specific_phenix_eff_file_path,
            "--phenix-out-mtz",
            phenix_out_name,
        ],
        check=True,
    )

    cmd = f"source {phenix_env} && cd {repo_dir} && phenix.refine {specific_phenix_eff_file_path} overwrite=True"  # {mtz_output_path} {r_free_path} {phenix_refine_eff_file_path} {pdb_model_path} overwrite=True"

    print("\nRunning phenix with command:", flush=True)
    print(cmd, flush=True)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
        )
        print("Phenix command completed successfully")

    except subprocess.CalledProcessError as e:
        print("\nPhenix.refine failed!", flush=True)
        print(f"Error code: {e.returncode}", flush=True)
        print("\nCommand output:", flush=True)
        print(e.stdout, flush=True)
        print("\nError output:", flush=True)
        print(e.stderr, flush=True)
        print("\nFull command that failed:", flush=True)
        print(cmd, flush=True)
        raise

    # if os.path.exists(specific_phenix_eff_file_path):
    #     os.remove(specific_phenix_eff_file_path)
    #     print(f"Deleted {specific_phenix_eff_file_path}")
    # else:
    #     print(f"{specific_phenix_eff_file_path} does not exist")

    return phenix_out_name


def find_peaks_from_model(model, repo_dir: str, checkpoint_path: str):
    surrogate_posterior_dataset = list(
        model.surrogate_posterior.to_dataset(only_observed=True)
    )[0]

    checkpoint_name, _ = os.path.splitext(os.path.basename(checkpoint_path))

    mtz_output_path = os.path.join(repo_dir, f"phenix_ready_{checkpoint_name}.mtz")

    print(f"\nWriting MTZ file to: {mtz_output_path}")
    surrogate_posterior_dataset.write_mtz(mtz_output_path)

    if os.path.exists(mtz_output_path):
        print(f"Successfully wrote MTZ file to: {mtz_output_path}")
    else:
        print("Error: Failed to write MTZ file")

    phenix_out_name = run_phenix(
        repo_dir=repo_dir,
        checkpoint_name=checkpoint_name,
        mtz_output_path=mtz_output_path,
    )
    # phenix_out_name = f"refine_{checkpoint_name}"

    # solved_artifact = wandb.Artifact("phenix_solved", type="mtz")

    pdb_file_path = glob.glob(os.path.join(repo_dir, f"{phenix_out_name}*.pdb"))
    print("pdb_file_path", pdb_file_path)
    pdb_file_path = pdb_file_path[-1]
    mtz_file_path = glob.glob(os.path.join(repo_dir, f"{phenix_out_name}*.mtz"))
    print("mtz_file_path", mtz_file_path)
    mtz_file_path = mtz_file_path[0]

    if pdb_file_path:
        print(f"Using PDB file: {pdb_file_path}")
        log_file_path = glob.glob(os.path.join(repo_dir, f"{phenix_out_name}*.log"))
        print("log_file_path", log_file_path)
        log_file_path = log_file_path[-1]
        print("log_file_path2", log_file_path)
        epoch_start, epoch_final, r_work, r_free = extract_r_values_from_log(
            log_file_path
        )
        print("rvals", r_work, r_free)
    else:
        print("No PDB files found matching pattern!")
        r_work, r_free = None, None

    # pdb_file_path = glob.glob(f"{phenix_out_name}*.pdb")[-1]
    # mtz_file_path = glob.glob(f"{phenix_out_name}*.mtz")[-1]
    print("run find peaks")
    print("mtz_file_path", mtz_file_path)
    print("pdb_file_path", pdb_file_path)
    print("repo_dir", repo_dir)
    peaks_file_path = os.path.join(repo_dir, f"peaks_{phenix_out_name}.csv")

    import reciprocalspaceship as rs

    ds = rs.read_mtz(mtz_file_path)
    print(ds.columns.tolist())

    subprocess.run(
        f"rs.find_peaks {mtz_file_path} {pdb_file_path} -f ANOM -p PANOM -z 5.0 -o {peaks_file_path}",
        shell=True,
        # cwd=f"{repo_dir}",
    )
    print("finished find peaks")
    return (
        epoch_start,
        r_work,
        r_free,
        os.path.join(repo_dir, f"peaks_{phenix_out_name}.csv"),
    )


def process_checkpoint(
    checkpoint_path, artifact_dir, model_settings, loss_settings, wandb_directory
):
    import os
    import re

    import pandas as pd
    from model import Model

    checkpoint_path_ = os.path.join(artifact_dir, checkpoint_path)
    print("checkpoint_path", checkpoint_path)
    print("checkpoint_path_", checkpoint_path_)
    print("artifact_dir", artifact_dir)

    try:
        match = re.search(r"best-(\d+)-", checkpoint_path)
        epoch = int(match.group(1)) if match else None
        model = Model.load_from_checkpoint(
            checkpoint_path, model_settings=model_settings, loss_settings=loss_settings
        )
        model.eval()

        epoch_start, r_work, r_free, path_to_peaks = find_peaks_from_model(
            model=model, repo_dir=wandb_directory, checkpoint_path=checkpoint_path
        )

        peaks_df = pd.read_csv(path_to_peaks)
        rows = []
        for _, peak in peaks_df.iterrows():
            residue = peak["residue"]
            peak_height = peak["peakz"]
            rows.append(
                {
                    "Epoch": epoch,
                    "Checkpoint": checkpoint_path,
                    "Residue": residue,
                    "Peak_Height": peak_height,
                }
            )
        return epoch_start, rows, r_work, r_free
    except Exception as e:
        print(f"Failed for {checkpoint_path}: {e}")
        return None, [], None, None


def run_phenix_over_all_checkpoints(
    model_settings,
    loss_settings,
    phenix_settings,
    artifact_dir,
    checkpoint_paths,
    wandb_directory,
):
    import multiprocessing as mp

    import pandas as pd
    import wandb

    all_rows = []
    r_works_start = []
    r_works_final = []
    r_frees_start = []
    r_frees_final = []
    max_peak_heights = []
    epochs = []

    ctx = mp.get_context("spawn")
    # Prepare argument tuples for each checkpoint
    args = [
        (checkpoint_path, artifact_dir, model_settings, loss_settings, wandb_directory)
        for checkpoint_path in checkpoint_paths
    ]
    print("start multithreading")
    with ctx.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_checkpoint, args)

    for epoch_start, rows, r_work, r_free in results:
        all_rows.extend(rows)
        if (
            isinstance(r_work, list)
            and len(r_work) == 2
            and isinstance(r_free, list)
            and len(r_free) == 2
            and r_work[0] is not None
            and r_work[1] is not None
            and r_free[0] is not None
            and r_free[1] is not None
        ):
            r_works_start.append(r_work[0])
            r_works_final.append(r_work[1])
            r_frees_start.append(r_free[0])
            r_frees_final.append(r_free[1])
        else:
            print("Malformed r_work or r_free:", r_work, r_free)

        if epoch_start is not None:
            epochs.append(epoch_start)
        try:
            max_peak_heights.append(max(row["Peak_Height"] for row in rows))
        except Exception as e:
            print(f"failed to find max peak height: {e}")
            max_peak_heights.append(0)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, max_peak_heights, marker="o", label="Max Peak Height")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Peak Height")
        ax.set_title("Max Peak Height vs Epoch")
        ax.legend()
        plt.tight_layout()
        wandb.log({"max_peak_heights_plot": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        print(f"failed to plot max peak heights: {e}")

    final_df = pd.DataFrame(all_rows)

    if not final_df.empty and "Peak_Height" in final_df.columns:
        wandb.summary["overall_largest_peak"] = final_df["Peak_Height"].max()
    else:
        wandb.summary["overall_largest_peak"] = 0

    wandb.log({"all_peak_heights_table": wandb.Table(dataframe=final_df)})
    if final_df.empty:
        print("No peak heights to log.")

    # def plot_peak_heights_vs_epoch(df):
    try:
        df = final_df
        import matplotlib.cm as cm

        if df.empty:
            print("No data to plot for peak heights vs epoch.")
            return
        plt.figure(figsize=(10, 6))
        residues = df["Residue"].unique()
        colormap = cm.get_cmap("tab20", len(residues))
        color_map = {res: colormap(i) for i, res in enumerate(residues)}
        for res in residues:
            sub = df[df["Residue"] == res]
            plt.scatter(
                sub["Epoch"], sub["Peak_Height"], label=str(res), color=color_map[res]
            )
        plt.xlabel("Epoch")
        plt.ylabel("Peak Height")
        plt.title("Peak Height vs Epoch per Residue")
        plt.legend(title="Residue", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        wandb.log({"PeakHeight_vs_Epoch": wandb.Image(plt.gcf())})
        plt.close()
        print("plot peak height vs epoch")

    except Exception as e:
        print(f"failed to plot peak heights: {e}")

    try:
        _, _, r_work_reference, r_free_reference = extract_r_values_from_log(
            phenix_settings.r_values_reference_path
        )
        r_work_reference = r_work_reference[-1]
        r_free_reference = r_free_reference[-1]
    except Exception as e:
        print(f"failed to extract reference r-vals: {e}")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        min_y = (
            min(
                min(r_works_final),
                min(r_frees_final),
                r_work_reference,
                r_free_reference,
            )
            - 0.01
        )
        max_y = (
            max(
                max(r_works_final),
                max(r_frees_final),
                r_work_reference,
                r_free_reference,
            )
            + 0.01
        )

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = epochs  # Arbitrary units (e.g., checkpoint index)
        # ax.plot(x, r_works_start, marker='o', label='R-work (start)', c="red", alpha=0.7)
        ax.plot(x, r_works_final, marker="o", label="R-work (final)", c="red")
        # ax.plot(x, r_frees_start, marker='s', label='R-free (start)', c="blue", alpha=0.7)
        ax.plot(x, r_frees_final, marker="s", label="R-free (final)", c="blue")

        if r_work_reference is not None:
            ax.axhline(
                r_work_reference,
                color="red",
                linestyle="dotted",
                linewidth=0.7,
                alpha=0.5,
                label="r_work_reference",
            )
        if r_free_reference is not None:
            ax.axhline(
                r_free_reference,
                color="blue",
                linestyle="dotted",
                linewidth=0.7,
                alpha=0.5,
                label="r_free_reference",
            )

        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("R Value")
        ax.set_ylim(min_y, max_y)
        ax.set_yticks(np.linspace(min_y, max_y, 12))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.4f"))
        ax.set_title("R-work and R-free")
        ax.legend(loc="best", fontsize=10, frameon=True)
        plt.tight_layout()
        wandb.log({"R_values_plot": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        print(f"failed plotting r-values: {e}")
