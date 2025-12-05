import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
directory = "./data/benchmark/"
df = pd.read_csv(directory + "benchmark_results.csv")

# Map integer modes to names for the chart
mode_map = {
    0: "CPU Serial",
    1: "CPU Parallel",
    2: "GPU Global",
    3: "GPU Global (Intrinsics)",
    4: "GPU Shared",
    5: "GPU Shared (Intrinsics)"
}
df['Mode Name'] = df['Mode'].map(mode_map)

# Set up the plot style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Map image IDs to actual sizes
size_map = {
    0: "64x64",
    1: "128x128",
    2: "256x256",
    3: "512x512"
}
df['Resolution'] = df['ImageID'].map(size_map)

# Create a bar chart with error bars (confidence interval)
sns.barplot(
    data=df, 
    x="Resolution", 
    y="Time_ms", 
    hue="Mode Name", 
    estimator=pd.Series.median,
    # errorbar='sd', # Show standard deviation
    errorbar=None,
    palette="viridis"
)

plt.title("Non-Local Means Performance Comparison")
plt.ylabel("Execution Time (ms)")
plt.xlabel("Image Resolution")
plt.yscale("log") # Log scale is vital because CPU is usually 100x slower
plt.legend(title="Implementation Mode", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(directory + "benchmark_plot.png")
print("Plot saved to " + directory + "benchmark_plot.png")
# plt.show()