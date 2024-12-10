import re
import pandas as pd

# Filepath for the log file
file_path = "execution_log.txt"

# Initialize an empty list to store extracted data
data = []

# Regular expressions to match the required lines
performer_type_pattern = re.compile(r"Using (.*?)-Performer now")
random_features_pattern = re.compile(r"Random feature number: (\d+)")
training_time_pattern = re.compile(r"Avg training speed per epoch: ([\d.]+)")
log_train_time_pattern = re.compile(r"Log_2\(T\) training speed: ([\d.-]+)")
log_inference_time_pattern = re.compile(r"Log_2\(T\) inference speed: ([\d.-]+)")
test_accuracy_pattern = re.compile(r"wandb: test_accuracy (\d+(?:\.\d+)?)")
run_name_pattern = re.compile(r"wandb: Syncing run (.+)")

# Read and process the log file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Temporary storage for each experiment
current_data = {}

for line in lines:
    performer_match = performer_type_pattern.search(line)
    random_features_match = random_features_pattern.search(line)
    training_time_match = training_time_pattern.search(line)
    log_train_time_match = log_train_time_pattern.search(line)
    log_inference_time_match = log_inference_time_pattern.search(line)
    test_accuracy_match = test_accuracy_pattern.search(line)
    run_name_match = run_name_pattern.search(line)

    if performer_match:
        if current_data:  # Save previous experiment's data if exists
            data.append(current_data)
        current_data = {"Performer-Type": performer_match.group(1)}
    
    if random_features_match:
        current_data["Random Features"] = int(random_features_match.group(1))
    
    if training_time_match:
        current_data["Training Time Per Epoch"] = float(training_time_match.group(1))
    
    if log_train_time_match:
        current_data["Log Train Time"] = float(log_train_time_match.group(1))
    
    if log_inference_time_match:
        current_data["Log Inference Time"] = float(log_inference_time_match.group(1))
    
    if test_accuracy_match:
        current_data["Test Accuracy"] = float(test_accuracy_match.group(1))
    
    if run_name_match:
        current_data["Run Name"] = run_name_match.group(1)

# Add the last experiment's data
if current_data:
    data.append(current_data)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save or display the formatted table
output_path = "formatted_output_with_run_names.csv"
df.to_csv(output_path, index=False)
print(f"Formatted data with run names saved to {output_path}")
