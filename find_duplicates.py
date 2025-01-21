import pandas as pd
from pathlib import Path

# Input file path
input_file = 'input/amazon_order_history_2024.csv'

# Generate output filename based on input filename
input_filename = Path(input_file).stem
output_file = f'output/{input_filename}_duplicates.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Find rows with duplicate order IDs
duplicates = df[df.duplicated(subset=['order id'], keep=False)]

# Sort by order id to group duplicates together
duplicates = duplicates.sort_values('order id')

# Save to output file
duplicates.to_csv(output_file, index=False)

# Print summary
print(f"\nFound {len(duplicates)} rows with duplicate order IDs")
print(f"Number of unique order IDs with duplicates: {len(duplicates['order id'].unique())}")
print(f"\nDuplicate rows have been saved to: {output_file}")
print("\nPreview of duplicate entries:")
print(duplicates[['order id', 'items', 'date', 'total']].to_string())