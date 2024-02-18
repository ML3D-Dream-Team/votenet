import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ['mAP@0.5', 'mAP@0.25', 'Mean Loss', 'Objectness Loss (scaled)']
values1 = [55.9, 31.2, 6.462984, 9.5]  # Values for the first column
values2 = [49.2, 22.1, 10.212211, 17.2]  # Values for the second column

# Calculate the width of each bar
bar_width = 0.35

# Set the x locations for the groups
x = np.arange(len(categories))

# Plot the bars
plt.bar(x - bar_width/2, values1, width=bar_width, label='Vanilla')
plt.bar(x + bar_width/2, values2, width=bar_width, label='Attention')

# Add labels, title, legend, etc.
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Vanilla vs Attention Votenet Mean Average Precision')
plt.xticks(x, categories)
plt.legend()
for i, (val1, val2) in enumerate(zip(values1, values2)):
    plt.text(i - bar_width/2, val1 + 0.5, str(val1), ha='center')
    plt.text(i + bar_width/2, val2 + 0.5, str(val2), ha='center')


# Show the plot
plt.show()