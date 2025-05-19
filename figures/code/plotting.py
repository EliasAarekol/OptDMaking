import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# # Assuming your CSV file is named 'data.csv' in the same directory
file_path = 'figures/crap/csvs/example.csv'
output_dir = 'rolling_plots'  # Directory to save the rolling window plots
window_size = 100  # Define the size of the rolling window

# import os
# os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# # Read the CSV file into a pandas DataFrame
# try:
#     df = pd.read_csv(file_path)
#     print("CSV file successfully read into a DataFrame:")
#     print(df.head())  # Display the first few rows of the DataFrame
# except FileNotFoundError:
#     print(f"Error: The file '{file_path}' was not found.")
#     exit()
# except Exception as e:
#     print(f"An error occurred while reading the CSV file: {e}")
#     exit()

# # Iterate through each column in the DataFrame and create a rolling window line plot
# for column in df.columns:
#     try:
#         # Apply a rolling window to the current column
#         rolling_mean = df[column].rolling(window=window_size).mean()

#         plt.figure(figsize=(12, 7))  # Adjust figure size as needed
#         # plt.plot(df.index, df[column], label='Original')
#         plt.plot(df.index, rolling_mean, label=f'Rolling Mean (Window={window_size})', color='red')
#         plt.xlabel('Index')
#         plt.ylabel(column)
#         plt.title(f'Rolling Window Line Plot of {column}')
#         plt.legend()
#         plt.grid(True)

#         # Save the figure as a PNG file
#         filename = os.path.join(output_dir, f'{column.replace(" ", "_")}_rolling_{window_size}.png')
#         plt.savefig(filename)
#         plt.close()  # Close the plot to free up memory
#         print(f"Rolling plot for '{column}' saved as '{filename}'")

#     except TypeError:
#         print(f"Warning: Skipping column '{column}' as it contains non-numeric data and cannot be plotted.")
#     except Exception as e:
#         print(f"An error occurred while plotting rolling window for column '{column}': {e}")

import os
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print("CSV file successfully read into a DataFrame:")
    print(df.head())  # Display the first few rows of the DataFrame
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    exit()

# Identify columns to keep (those that don't contain "__MIN" or "__MAX")
columns_to_keep = [col for col in df.columns if "__MIN" not in col and "__MAX" not in col and col != "Step"]

if not columns_to_keep:
    print("No suitable columns found for calculating the mean after dropping min/max columns.")
    exit()

# Select the columns to calculate the mean from
df_mean = df[columns_to_keep]

# Calculate the mean across the selected columns
try:
    df['mean_reward'] = df_mean.mean(axis=1)

    # Apply a rolling window to the calculated mean
    rolling_mean = df['mean_reward'].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 7))  # Adjust figure size as needed
    plt.plot(df['Step'], df['mean_reward'], label='Mean Reward')
    plt.plot(df['Step'], rolling_mean, label=f'Rolling Mean (Window={window_size})', color='red')
    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.title(f'Rolling Window of Mean Reward (excluding MIN/MAX columns)')
    plt.legend()
    plt.grid(True)

    # Format the x-axis to display in thousands (k)
    def format_thousands(x, pos):
        return f'{x/1000:.0f}k'

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))

    # Save the figure as a PNG file
    filename = os.path.join(output_dir, f'mean_reward_rolling_{window_size}_k_xaxis.png')
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory
    print(f"Rolling plot of mean reward (k-axis) saved as '{filename}'")

except TypeError:
    print("Error: One or more of the selected columns contain non-numeric data and cannot be used for calculating the mean.")
except Exception as e:
    print(f"An error occurred while calculating or plotting the mean reward: {e}")

# Plot all normalized columns on the same plot
try:
    plt.figure(figsize=(12, 7))  # Create a single figure

    for column in columns_to_keep:
        # Normalize the column data
        col_data = df[column]
        normalized_data = (col_data - col_data.min()) / (col_data.max() - col_data.min())  # Normalize to [0, 1]
        normalized_data = normalized_data.rolling(window = window_size).mean()
        plt.plot(df['Step'], normalized_data, label=column)  # Plot on the same figure

    plt.xlabel('Step')
    plt.ylabel('Normalized Value')
    plt.title('Normalized Plot of All Columns')
    plt.grid(True)
    plt.legend()

    # Format the x-axis to display in thousands (k)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))

    filename = os.path.join(output_dir, 'all_columns_normalized_k_xaxis.png')
    plt.savefig(filename)
    plt.close()
    print(f"Combined normalized plot saved as '{filename}'")

except TypeError:
    print("Error: One or more columns contain non-numeric data and cannot be plotted.")
except Exception as e:
    print(f"An error occurred while plotting normalized columns: {e}")

print(f"\nAll normalized plots have been saved to the '{output_dir}' directory.")


normalized_df=(df-df.min())/(df.max()-df.min())



df_mean = normalized_df[columns_to_keep]

# Calculate the mean across the selected columns
try:
    df['mean_reward'] = df_mean.mean(axis=1)

    # Apply a rolling window to the calculated mean
    rolling_mean = df['mean_reward'].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 7))  # Adjust figure size as needed
    plt.plot(df['Step'], df['mean_reward'], label='Mean Reward')
    plt.plot(df['Step'], rolling_mean, label=f'Rolling Mean (Window={window_size})', color='red')
    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.title(f'Rolling Window of Mean Reward (excluding MIN/MAX columns)')
    plt.legend()
    plt.grid(True)

    # Format the x-axis to display in thousands (k)
    def format_thousands(x, pos):
        return f'{x/1000:.0f}k'

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_thousands))

    # Save the figure as a PNG file
    filename = os.path.join(output_dir, f'norm_mean.png')
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory
    print(f"Rolling plot of mean reward (k-axis) saved as '{filename}'")

except TypeError:
    print("Error: One or more of the selected columns contain non-numeric data and cannot be used for calculating the mean.")
except Exception as e:
    print(f"An error occurred while calculating or plotting the mean reward: {e}")
