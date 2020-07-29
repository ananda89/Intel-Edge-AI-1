import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_style("whitegrid")
sns.set_palette("muted")

output_dir = "images/"

models = [
    "face_detection",
    "facial_landmarks_detection",
    "head_pose_estimation",
    "gaze_estimation",
]

# data for FP32 precision
model_load_times_FP32 = [389.33, 122.49, 115.23, 135.79]
input_processing_times_FP32 = [0.995, 0.055, 0.062, 0.056]
inference_time_FP32 = 93.4
fps_FP32 = 6.32

# data for FP16 precision
model_load_times_FP16 = [389.75, 120.40, 129.49, 144.70]
input_processing_times_FP16 = [1.018, 0.052, 0.061, 0.051]
inference_time_FP16 = 93.54
fps_FP16 = 6.31

# data for INT8 precision
model_load_times_INT8 = [650.12, 148.93, 257.29, 290.51]
input_processing_times_INT8 = [0.980, 0.049, 0.058, 0.050]
inference_time_INT8 = 93.34
fps_INT8 = 6.32

df = pd.DataFrame(
    list(
        zip(
            model_load_times_FP32,
            model_load_times_FP16,
            model_load_times_INT8,
            input_processing_times_FP32,
            input_processing_times_FP16,
            input_processing_times_INT8,
        )
    ),
    index=models,
    columns=[
        "load_time_FP32",
        "load_time_FP16",
        "load_time_INT8",
        "process_time_FP32",
        "process_time_FP16",
        "process_time_INT8",
    ],
)
print(df)
"""
Plot 1: Model load times for different models with different precisions
"""
# df[["load_time_FP32", "load_time_FP16", "load_time_INT8"]].plot(
#     kind="bar", alpha=0.6
# )
# plt.xticks(rotation=0)
# plt.xlabel("model name", fontsize=14)
# plt.ylabel("time (in ms)", fontsize=14)
# plt.title("Model load times for various precisions", fontsize=16)
# plt.legend(["FP32", "FP16", "INT8"], fontsize=14)

"""
Plot 2: Time taken to preprocess inputs for various models
"""
# df[["process_time_FP32", "process_time_FP16", "process_time_INT8"]].plot(
#     kind="bar", alpha=0.6
# )
# plt.xticks(rotation=0)
# plt.xlabel("model name", fontsize=14)
# plt.ylabel("time (in ms)", fontsize=14)
# plt.title(
#     "Time taken to preprocess inputs for various models", fontsize=16,
# )
# plt.legend(["FP32", "FP16", "INT8"], fontsize=14)

"""
Plot 3: Plot of inference times for various precisions
"""
# plt.bar(["FP32", "FP16", "INT8"], [93.4, 93.54, 93.34], alpha=0.6)
# plt.ylim((93.0, 93.6))
# plt.xlabel("Precision", fontsize=14)
# plt.ylabel("Inference time (in sec)", fontsize=14)
# plt.title("Inference time on video feed", fontsize=16)

"""
Plot 4: Plot of fps for various precisions
"""
plt.bar(["FP32", "FP16", "INT8"], [6.32, 6.31, 6.32], alpha=0.6)
plt.ylim((6.3, 6.33))
plt.xlabel("Precision", fontsize=14)
plt.ylabel("fps (in sec)", fontsize=14)
plt.title("Frames per second on video feed", fontsize=16)

plt.show()
