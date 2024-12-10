import subprocess
import os

# Define the commands to run
commands = [
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_exp_8.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_relu_8.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_exp_8.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_relu_8.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_exp_8.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_relu_8.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_softmax_8.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_exp_16.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_relu_16.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_exp_16.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_relu_16.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_exp_16.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_relu_16.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_softmax_16.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_exp_32.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_relu_32.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_exp_32.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_relu_32.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_exp_32.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_relu_32.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_softmax_32.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_exp_64.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_relu_64.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_exp_64.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_relu_64.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_exp_64.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_relu_64.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_softmax_64.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_exp_4.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln1_relu_4.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_exp_4.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_ln2_relu_4.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_exp_4.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_relu_4.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_softmax_4.yaml',
    r'C:/Users/xxiao/.conda/envs/vit/python.exe c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/vit_performer.py --config c:/Users/xxiao/Downloads/Xiao/ViT-Performer/vit_performer/config/config_cifar10_trans.yaml',
]

# Log file path
log_file = "execution_log.txt"

# Open the log file for writing
with open(log_file, "w") as log:
    for i, command in enumerate(commands, start=1):
        log.write(f"Running command {i}: {command}\n")
        log.write("=" * 80 + "\n")
        
        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Log the output
            log.write("Output:\n")
            log.write(result.stdout + "\n")
            
            # Log any errors
            if result.stderr:
                log.write("Errors, warnings, messages:\n")
                log.write(result.stderr + "\n")
            
            # Log the exit code
            log.write(f"Exit code: {result.returncode}\n")
        except Exception as e:
            # Log any exceptions raised during execution
            log.write(f"Exception occurred while running command {i}: {e}\n")
        
        log.write("=" * 80 + "\n\n")

print(f"Execution completed. Logs are saved in {os.path.abspath(log_file)}")
