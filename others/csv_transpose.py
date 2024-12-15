import os
import pandas as pd

def transpose_csv_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # Read the CSV file
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)

            # Transpose the data
            transposed_df = df.transpose()

            # Create the output file path
            output_file_path = os.path.join(output_folder, filename)

            # Save the transposed data to CSV
            transposed_df.to_csv(output_file_path, index=False)

            print(f"Transposed file saved: {output_file_path}")

# Example usage
input_folder = 'C:/Users/yvona/Documents/NPU_research/research/SSVS/attentions/ODS'
output_folder = 'C:/Users/yvona/Documents/NPU_research/research/SSVS/attentions/ODS_transpose'

transpose_csv_files(input_folder, output_folder)
