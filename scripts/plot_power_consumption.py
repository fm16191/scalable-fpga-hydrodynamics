#!/bin/env python3

import re
import plotly.graph_objects as go
import numpy as np
from sys import argv
import scipy.signal
import os

if len(argv) < 2:
    print(f"Usage {argv[0]} file.txt")
    exit(1)

file = argv[1]
print(f"Processing file: {file}")

with open(file, "r") as fp:
    lines = fp.readlines()

timestamps = []
total_input_power = []
total_fpga_power = []

time_pattern = re.compile(r"^([\d\.]+) Total Input Power.*")
input_power_pattern = re.compile(r"Total Input Power\s+([\d\.]+) Watts")
fpga_power_pattern = re.compile(r"Total FPGA Power\s+([\d\.]+) Watts")

# Parse the log lines
i = 0
print("Parsing log lines...", end='\r')
while i < len(lines):
    time_match = time_pattern.search(lines[i])
    input_power_match = input_power_pattern.search(lines[i])
    
    if not input_power_match:
        i += 1
        continue

    if time_match:
        timestamps.append(float(time_match.group(1)))
    total_input_power.append(float(input_power_match.group(1)))

    if i + 1 < len(lines):  
        fpga_power_match = fpga_power_pattern.search(lines[i + 1])
        if fpga_power_match:
            total_fpga_power.append(float(fpga_power_match.group(1)))

    i += 1

if not total_input_power or not total_fpga_power:
    print("No data found for power consumption.")
    exit(1)

print(f"Data points : {len(total_input_power):.2f}, {len(total_fpga_power):.2f}")

# Apply median filter to smooth power values (optional, commented out here)
# smoothed_input_power = scipy.signal.medfilt(total_input_power, kernel_size=11)
# smoothed_fpga_power = scipy.signal.medfilt(total_fpga_power, kernel_size=11)

# Compute stable mean from the first 80% of values
keep_factor=1
stable_mean_input = np.mean(total_input_power[:int(len(total_input_power) * keep_factor)])
stable_mean_fpga = np.mean(total_fpga_power[:int(len(total_fpga_power) * keep_factor)])

print(f"Stable means for {keep_factor*100}% of values : {stable_mean_input}, {stable_mean_fpga}")

tdp_limit = 90

if total_input_power and total_fpga_power:
    max_value = max(max(total_input_power), max(total_fpga_power), tdp_limit)
else:
    max_value = tdp_limit

print("Generating plot...", end='\r')

fig = go.Figure()

fig.add_trace(go.Scatter(x=timestamps, y=total_input_power, mode='lines', name='Total Input Power', line=dict(color='blue')))

fig.add_trace(go.Scatter(x=timestamps, y=total_fpga_power, mode='lines', name='Total FPGA Power', line=dict(color='green')))

fig.add_trace(go.Scatter(x=timestamps, y=[stable_mean_input] * len(timestamps), mode='lines', name='Mean Total Input Power', line=dict(dash='dash', color='red')))
fig.add_trace(go.Scatter(x=timestamps, y=[stable_mean_fpga] * len(timestamps), mode='lines', name='Mean Total FPGA Power', line=dict(dash='dash', color='orange')))

fig.add_trace(go.Scatter(x=timestamps, y=[tdp_limit] * len(timestamps), mode='lines', name='TDP Limit (90W)', line=dict(dash='dot', color='purple')))


fig.add_annotation(
    x=1.28,
    y=stable_mean_input,
    xref="paper",
    yref="y",
    text=f'Mean Total Input Power: {stable_mean_input:.2f} W',
    showarrow=False,
    font=dict(size=12, color='black'),
    align="left"
)

fig.add_annotation(
    x=1.28,
    y=stable_mean_fpga,
    xref="paper",
    yref="y",
    text=f'Mean Total FPGA Power: {stable_mean_fpga:.2f} W',
    showarrow=False,
    font=dict(size=12, color='black'),
    align="left"
)

fig.update_layout(title='FPGA Power Consumption Over Time',
                  xaxis_title='Time (seconds)',
                  yaxis_title='Power (W)',
                #   template='plotly_dark',
                  yaxis=dict(range=[0, max_value]))

# Extracting file name without extension for output
filename = os.path.splitext(os.path.basename(file))[0]
output_filename = f"fpga_power_consumption_{filename}"

# fig.write_image(f"{output_filename}.png", width=1920//2, height=1080//2, scale=2)
fig.write_image(f"{output_filename}.svg", width=1920//2, height=1080//2, scale=2)

print(f"Plot saved successfully : {output_filename}.{{svg,png}}")
