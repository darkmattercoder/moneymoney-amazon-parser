#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path
import os
from datetime import datetime
import re

def create_work_directory(input_file: str) -> str:
    """
    Create a work directory based on input filename and timestamp.
    Args:
        input_file (str): Path to the input CSV file
    Returns:
        str: Path to the created work directory
    """
    # Get input filename without extension
    input_filename = Path(input_file).stem

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # Create work directory path
    work_dir = os.path.join('process', f"{input_filename}_{timestamp}")

    # Create directories if they don't exist
    os.makedirs(work_dir, exist_ok=True)

    return work_dir

def format_statistics(df: pd.DataFrame) -> list:
    """
    Format statistics for better readability.
    Args:
        df: pandas DataFrame
    Returns:
        list: Formatted statistics lines
    """
    stats = []
    desc = df.describe()

    # For each column
    for column in desc.columns:
        stats.append(f"\nStatistics for column: {column}")
        stats.append("-" * 40)

        # For each statistic
        for stat_name, stat_value in desc[column].items():
            # Format numbers with appropriate precision
            if isinstance(stat_value, (int, float)):
                if stat_value.is_integer():
                    formatted_value = f"{int(stat_value):,}"
                else:
                    formatted_value = f"{stat_value:,.2f}"
            else:
                formatted_value = str(stat_value)

            stats.append(f"{stat_name:>8}: {formatted_value}")

    return stats

def extract_order_id_from_url(url: str) -> str:
    """
    Extract order ID from Amazon URL.
    Args:
        url: Amazon order URL
    Returns:
        str: Extracted order ID or empty string if not found
    """
    if pd.isna(url):
        return ""

    # First try the order ID from the URL parameter
    match = re.search(r'orderID=([A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+)', url)
    if match:
        return match.group(1)

    # If not found, try to get it from the order id column
    # This is needed because some URLs might have a different format
    return ""

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the DataFrame, considering URLs with same order ID as duplicates.
    For order IDs starting with 'D01-', the invoice column is ignored in the comparison.
    Args:
        df: pandas DataFrame
    Returns:
        tuple: (deduplicated DataFrame, number of duplicates removed)
    """
    # Get initial row count
    initial_count = len(df)

    # Create a copy of the DataFrame
    df_work = df.copy()

    # Get all columns except 'order url' for comparison
    compare_cols = [col for col in df.columns if col != 'order url']

    # Create a list to store indices to keep
    indices_to_keep = []
    seen_combinations = set()

    # Iterate through rows to find unique combinations
    for idx, row in df_work.iterrows():
        # Use the order ID from the order id column
        order_id = row['order id']

        # For order IDs starting with 'D01-', exclude 'invoice' from comparison
        if order_id.startswith('D01-'):
            comparison_cols = [col for col in compare_cols if col != 'invoice']
        else:
            comparison_cols = compare_cols

        # Create a tuple of values for comparison (excluding original URL and possibly invoice)
        comparison_values = tuple(row[comparison_cols].values)

        # Create a unique key for this row
        row_key = (order_id,) + comparison_values

        # If we haven't seen this combination before, keep it
        if row_key not in seen_combinations:
            indices_to_keep.append(idx)
            seen_combinations.add(row_key)

    # Create deduplicated DataFrame
    df_deduped = df.loc[indices_to_keep].copy()

    # Calculate number of duplicates removed
    duplicates_removed = initial_count - len(df_deduped)

    return df_deduped, duplicates_removed

def count_items(items_str: str) -> int:
    """
    Count the number of items in an items string.
    Args:
        items_str: String containing items separated by semicolons
    Returns:
        int: Number of items
    """
    if pd.isna(items_str):
        return 0
    # Split by semicolon and filter out empty strings
    items = [item.strip() for item in items_str.split(';') if item.strip()]
    return len(items)

def is_valid_order_row(row: pd.Series) -> bool:
    """
    Check if a row contains actual order data.
    Args:
        row: pandas Series representing a row
    Returns:
        bool: True if the row contains actual order data, False otherwise
    """
    # First check if any field contains "SUBTOTAL"
    for field in row:
        if pd.notna(field) and "SUBTOTAL" in str(field).upper():
            return False

    # Check key fields that should be present in a valid order
    key_fields = ['order id', 'items', 'date']

    for field in key_fields:
        if pd.notna(row[field]) and str(row[field]).strip() not in ['', ',', ';']:
            return True
    return False

def process_csv(input_file: str):
    """
    Process the input CSV file.
    Args:
        input_file (str): Path to the input CSV file
    """
    try:
        # Create work directory
        work_dir = create_work_directory(input_file)

        # Read the CSV file
        df = pd.read_csv(input_file)

        # Check if last row is invalid and remove it if necessary
        original_row_count = len(df)
        if len(df) > 0 and not is_valid_order_row(df.iloc[-1]):
            df = df.iloc[:-1]
            print(f"Removed last row as it did not contain valid order data")

        # Count total items
        total_items = df['items'].apply(count_items).sum()

        # Prepare analytics output
        analytics = []
        analytics.append(f"CSV File Analysis Report")
        analytics.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analytics.append(f"\nInput File: {input_file}")
        if original_row_count != len(df):
            analytics.append(f"Original row count: {original_row_count:,}")
            analytics.append(f"Rows after removing invalid last row: {len(df):,}")
        analytics.append(f"Number of rows: {len(df):,}")
        analytics.append(f"Number of columns: {len(df.columns):,}")
        analytics.append(f"Total number of items: {total_items:,}")
        analytics.append("\nColumns:")
        for col in df.columns:
            analytics.append(f"- {col}")

        # Add formatted statistics
        analytics.append("\nDetailed Statistics:")
        analytics.extend(format_statistics(df))

        # Add items per row statistics
        items_per_row = df['items'].apply(count_items)
        analytics.append("\nItems per Row Analysis:")
        analytics.append("-" * 40)
        analytics.append(f"Average items per row: {items_per_row.mean():.2f}")
        analytics.append(f"Maximum items in a row: {items_per_row.max():,}")
        analytics.append(f"Minimum items in a row: {items_per_row.min():,}")
        analytics.append(f"Rows with multiple items: {len(items_per_row[items_per_row > 1]):,}")
        analytics.append(f"Rows with single item: {len(items_per_row[items_per_row == 1]):,}")
        analytics.append(f"Rows with no items: {len(items_per_row[items_per_row == 0]):,}")

        # Remove duplicates and save deduplicated data
        df_deduped, duplicates_removed = remove_duplicates(df)
        deduped_file = os.path.join(work_dir, 'data_deduped.csv')
        df_deduped.to_csv(deduped_file, index=False)

        # Add deduplication information to analytics
        analytics.append(f"\nDuplication Analysis:")
        analytics.append("-" * 40)
        analytics.append(f"Original row count: {len(df):,}")
        analytics.append(f"Rows after deduplication: {len(df_deduped):,}")
        analytics.append(f"Duplicate rows removed: {duplicates_removed:,}")
        if duplicates_removed > 0:
            analytics.append(f"Duplication rate: {(duplicates_removed/len(df)*100):.2f}%")

        # Write analytics to file
        metadata_file = os.path.join(work_dir, 'input_metadata')
        with open(metadata_file, 'w') as f:
            f.write('\n'.join(analytics))

        # Also print to console
        print('\n'.join(analytics))
        print(f"\nAnalytics have been saved to: {metadata_file}")
        print(f"Deduplicated data has been saved to: {deduped_file}")

    except Exception as e:
        print(f"Error processing the CSV file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process a CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()

    # Check if file exists
    if not Path(args.input_file).is_file():
        print(f"Error: File '{args.input_file}' does not exist")
        return

    # Process the CSV file
    process_csv(args.input_file)

if __name__ == "__main__":
    main()