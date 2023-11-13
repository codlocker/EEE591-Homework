import numpy as np
import subprocess

import os
import re
from utils import create_directory, delete_all_files_in_dir, validate_paths

FAN_RANGE = 9
NO_OF_INVERTERS = 9

# Generate fan numbers and inverter numbers
# from max count
#    Args:
#        fans (int): the max count
#        invs (int): the max count

#    Returns:
#        list: list of fan numbers and inverter numbers
def generate_fan_inv_list(fans: int, invs: int) -> list:
    # Get the fans count
    fans = [f_idx + 2 for f_idx in range(fans)]

    # Generate invertors list
    inverters = [inv for inv in range(invs * 2) if inv % 2 != 0]

    inverters.remove(1)

    return fans, inverters

# Creates the new set in the hspice file format.
# Args:
#        fan (int): the number of fans in the circuit.
#        inverters (int): number of inverters in the circuit

# Returns:
#     str: Return the string for hspice file.
def create_hspice_format(fan: int, inverters: int) -> str:
    fan_param = f".param fan = {fan}\n"
    cap_load = f".param capacitance = 30pF\n"
    inv_content = "Xinv1 a b inv M=1\n"
    last_index = 0
    for inv in range(inverters - 2):
        inv_content += f"Xinv{str(inv + 2)} {chr(97 + (inv + 1))} {chr(97 + (inv + 2))} inv M=fan**{str(inv + 1)}\n"
        last_index = inv + 1

    if inverters > 1:
        inv_content += f"Xinv{str(last_index + 2)} {chr(97 + (last_index + 1))} z inv M=fan**{str(last_index + 1)}"
    
    return fan_param + cap_load + inv_content

############################################################
# Run hspice to generate the required output in csv format #
############################################################

# launch hspice. Note that both stdout and stderr are captured so
# they do NOT go to the terminal!
# Args:
#    input (str): Input file path
#    output (str): Out file path
def run_hspice(input: str, output: str):
    proc = subprocess.Popen(["hspice", "-i", input, "-o", output],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output, err = proc.communicate()
    print(err)

if __name__ == "__main__":
    ## 0. Create a directory for inputs and validate if all files exists in the folder
    # 0.1 Validate whether required files in the path of python program execution.
    print('Validating required files exist....')
    if not validate_paths(['cmoslibrary.lib', 'InvChainTemplate.sp', 'utils.py']):
        exit(1)
    
    print('Creating input and output directories....')
    input_files = './input/'
    output_files = './output/'
    # 0.2 Create the input and output directories for storage
    create_directory(input_files)
    create_directory(output_files)

    # 0.3 Delete all existing files in that path to avoid overwriting issues.
    delete_all_files_in_dir(input_files)
    delete_all_files_in_dir(output_files)

    ## 1. Get current content of file
    template_str = ""
    with open('InvChainTemplate.sp') as f:
        template_str = f.read()

    ## 2. Get the fans and inverter count
    fans, inverters = generate_fan_inv_list(FAN_RANGE, NO_OF_INVERTERS)
    print(fans, inverters)

    ## 3. Now create new content for each combination
    minimum_delays = []
    for fan in fans:
        print(f"========== Fan : {fan} ==========\n")

        for inverter in inverters:
            input_file_name = f'InvChain_{fan}_{inverter}.sp'
            output_file_name = f'InvChain_{fan}_{inverter}'
            with open(f"{input_files}{input_file_name}", "w+") as f:
                new_content = re.sub(
                    pattern=r'<.*>',
                    repl=create_hspice_format(fan, inverter), 
                    string=template_str)
                f.write(new_content)

            run_hspice(f"{input_files}{input_file_name}", f"{output_files}{output_file_name}")

            output_file = f"{output_files}{output_file_name}.mt0.csv"
            if os.path.isfile(output_file):
                data = np.recfromcsv(output_file, comments="$", skip_header=3)

                # Get tphl value
                tphl = data['tphl_inv']

                print(f'Delay for {inverter} inverters = {tphl} seconds')
                minimum_delays.append((tphl, fan, inverter))
            else:
                print(f'Could not find file with name {output_file}')

    # 4. Find the fan speed and inverter for the minimum delay value
    print("==========RESULT============")
    lowest_delay = min(minimum_delays)

    print(f"Lowest delay = {lowest_delay[0]}")
    print(f"Fan speed at lowest delay = {lowest_delay[1]}")
    print(f"No. of inverters at lowest delay = {lowest_delay[2]}")