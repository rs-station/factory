#!/usr/bin/env dials.python
def run_dials(dials_env, command):
    full_command = f"source {dials_env} && {command}"

    try:
        result = subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        # Print more detailed error messages
        print(f"Command failed with error code: {e.returncode}")
        print("Standard Output (stdout):")
        print(e.stdout if e.stdout else "No stdout output")
        print("Standard Error (stderr):")
        print(e.stderr if e.stderr else "No stderr output")
        raise


def run():
    refl_file="/n/hekstra_lab/people/aldama/subset/small_dataset/pass1/reflections_.refl"
    scale_command = (
        f"dials.scale '{refl_file}' '{expt_file}' "
        f"output.reflections='{scaled_refl_out}' "
        f"output.experiments='{scaled_expt_out}' "
        f"output.html='{parent_dir}/dials_out/scaling.html' "
        f"output.html='{output_dir}/scaling.html' "
        f"output.log='{output_dir}/scaling.log' "
    )
    print("Executing scale command:", scale_command)
    dials_env = "/n/hekstra_lab/people/aldama/software/dials-v3-16-1/dials_env.sh"
    run_dials(dials_env, scale_command)

if name==main:
    run()