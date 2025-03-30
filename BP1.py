import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------------
# New auth function using BP2 coding
# -------------------------------
def auth(training_file_paths, testing_file_paths, used_device, TEST_N):
    """
    Authentication function modified to use BP2 (backpropagation neural network)
    coding rather than DistilBERT. For each dataset, the function trains a simple
    neural network (LocationPredictor) and then tests it to produce a confusion table.
    """

    # BP2's load_dataset: reads CSV with (time, x, y, z) and converts features and targets
    def load_dataset(file_path, target_user):
        df = pd.read_csv(file_path, header=None, names=['time', 'x', 'y', 'z'])
        data = torch.tensor(df[['x', 'y', 'z']].values, dtype=torch.float32)
        targets = torch.ones((len(data), 1), dtype=torch.float32) if target_user else torch.zeros((len(data), 1), dtype=torch.float32)
        return TensorDataset(data, targets), df

    # BP2's simple neural network model
    class LocationPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LocationPredictor, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()  # For binary classification
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    # Hyperparameters (from BP2 coding)
    input_size = 3       # x, y, z
    hidden_size = 2      # two neurons in the hidden layer
    output_size = 1      # binary output (target user vs. others)
    learning_rate = 0.001
    epochs = 16
    batch_size = 32

    print("Training and validating for User 1...")
    u = 0  # index for labeling

    # -------------------------------
    # Training loop: one model per training file
    # -------------------------------
    for u in range(len(training_file_paths)):
        print(f"\n=== Processing Authentication for User {u+1} ===")
        for i, file_path in enumerate(training_file_paths):
            dataset, _ = load_dataset(file_path, target_user=(i == u))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model = LocationPredictor(input_size, hidden_size, output_size)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            for epoch in range(epochs):
                total_loss = 0.0
                for inputs, targets in data_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                #print(f"Dataset {i+1}, Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")
            
            model_save_path = f"D:/Spring semester 2025/MLS/Models/location_predictor_model_user{i+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            #print(f"Model for dataset {i+1} saved as '{model_save_path}'.")

        # -------------------------------
        # Testing loop: evaluate models on testing files and compile confusion table
        # -------------------------------
        
                # -------------------------------
        # Improved Testing loop: evaluate models on testing files with repeated sampling and aggregate metrics
        # -------------------------------
        num_iterations = 20
        metrics_list = []
        
        for iteration in range(num_iterations):
            all_outputs = []
            all_targets = []
            for i, file_path in enumerate(testing_file_paths):
                # Load the testing dataset (with balanced target sampling)
                dataset, _ = load_dataset(file_path, target_user=(i == u))
                # Define sample size: full TEST_N for target user; for others use a smaller sample (to address imbalance)
                sample_size = TEST_N if (i == u) else int(TEST_N / 16)
                sample_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
                sampled_data = torch.stack([dataset[idx][0] for idx in sample_indices])
                sampled_targets = torch.stack([dataset[idx][1] for idx in sample_indices])
                
                # Load the corresponding saved model
                model = LocationPredictor(input_size, hidden_size, output_size)
                model_load_path = f"D:/Spring semester 2025/MLS/Models/location_predictor_model_user{i+1}.pth"
                state_dict = torch.load(model_load_path)
                model.load_state_dict(state_dict)
                model.eval()
                
                with torch.no_grad():
                    outputs = model(sampled_data)
                    all_outputs.extend(outputs.cpu().numpy().flatten().tolist())
                    all_targets.extend(sampled_targets.cpu().numpy().flatten().tolist())
            
            # Compute confusion matrix components using threshold 0.5
            outputs_arr = np.array(all_outputs)
            targets_arr = np.array(all_targets)
            predictions = (outputs_arr >= 0.5).astype(int)
            
            tp = ((predictions == 1) & (targets_arr == 1)).sum()
            tn = ((predictions == 0) & (targets_arr == 0)).sum()
            fp = ((predictions == 1) & (targets_arr == 0)).sum()
            fn = ((predictions == 0) & (targets_arr == 1)).sum()
            
            # Compute performance metrics
            fnr = 100 * fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = 100 * fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = 100 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1_score = 100 * (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            try:
                aur = roc_auc_score(targets_arr, outputs_arr) * 100
            except Exception:
                aur = 0.0
            
            try:
                fpr_vals, tpr_vals, _ = roc_curve(targets_arr, outputs_arr)
                fnr_vals = 1 - tpr_vals
                abs_diffs = np.abs(fpr_vals - fnr_vals)
                min_index = np.argmin(abs_diffs)
                eer = ((fpr_vals[min_index] + fnr_vals[min_index]) / 2) * 100
            except Exception:
                eer = 0.0
            
            metrics_list.append([tp, tn, fp, fn, fnr, fpr, tpr, tnr, accuracy, f1_score, aur, eer])
        
        # Compute average and standard deviation of the metrics over iterations
        metrics_array = np.array(metrics_list)
        avg_metrics = np.mean(metrics_array, axis=0)
        std_metrics = np.std(metrics_array, axis=0)
        
        metric_names = ["TP", "TN", "FP", "FN", "FNR", "FPR", "TPR", "TNR", "Accuracy", "F1-score", "AUR", "EER%"]
        
        # Create DataFrame for each iteration's metrics
        iterations = list(range(1, num_iterations + 1))
        iteration_results = pd.DataFrame(metrics_list, columns=metric_names)
        iteration_results.insert(0, "Iteration", iterations)
        
        # Prepare a summary table with averages and standard deviations
        summary_results = pd.DataFrame([
            ["Average"] + list(avg_metrics),
            ["Std Dev"] + list(std_metrics)
        ], columns=["Metric"] + metric_names)
        
        # Write both DataFrames to Excel in separate sheets
        excel_path = f"D:/Spring semester 2025/MLS/Results/BP/confusion_table_BP_{used_device}{u+1}.xlsx"
        with pd.ExcelWriter(excel_path) as writer:
            iteration_results.to_excel(writer, sheet_name="Iteration Metrics", index=False)
            summary_results.to_excel(writer, sheet_name="Summary", index=False)
        print(f"Confusion table with iteration metrics and summary saved to {excel_path}")




# --------------------------------------------------
# Below, the file paths are set for different devices,
# and the modified auth function is called accordingly.
# --------------------------------------------------

# For Samsung
used_device = 'SAM'
TEST_N = 5000   # Number of samples for the target user in testing; others get TEST_N/16
training_file_paths = [
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User001/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User002/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User003/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User004/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User005/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User006/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User007/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User008/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User009/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0010/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0011/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0012/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0013/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0014/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0015/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0016/Samsung - back/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0017/Samsung - back/accelDataM.txt"
]
testing_file_paths = [p.replace("Training Data", "Testing Data") for p in training_file_paths]
auth(training_file_paths, testing_file_paths, used_device, TEST_N)

# For HTC
used_device = 'HTC'
TEST_N = 5000   # Number of samples for the target user in testing; others get TEST_N/16
training_file_paths = [
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User001/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User002/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User003/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User004/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User005/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User006/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User007/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User008/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User009/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0010/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0011/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0012/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0013/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0014/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0015/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0016/HTC - front/accelDataM.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0017/HTC - front/accelDataM.txt"
]
testing_file_paths = [p.replace("Training Data", "Testing Data") for p in training_file_paths]
auth(training_file_paths, testing_file_paths, used_device, TEST_N)

# For Google Glasses
used_device = 'GOO'
TEST_N = 704   # Adjusted sample size for target user; others get TEST_N/16
training_file_paths = [
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User001/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User002/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User003/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User004/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User005/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User006/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User007/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User008/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User009/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0010/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0011/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0012/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0013/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0014/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0015/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0016/Glass/accelData.txt",
    "D:/Spring semester 2025/MLS/First Data set/data/Training Data/User0017/Glass/accelData.txt"
]
testing_file_paths = [p.replace("Training Data", "Testing Data") for p in training_file_paths]
auth(training_file_paths, testing_file_paths, used_device, TEST_N)
