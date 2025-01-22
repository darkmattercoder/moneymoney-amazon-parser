#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path
import os
from datetime import datetime
import re
import sys

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

def get_items_file_path(orders_file: str) -> str:
    """
    Get the path to the items file based on the orders file path.
    Args:
        orders_file (str): Path to the orders file
    Returns:
        str: Path to the items file
    """
    orders_path = Path(orders_file)
    items_file = orders_path.parent / f"amazon_item_history_{orders_path.stem.split('_')[-1]}.csv"
    return str(items_file)

def extract_items_from_orders(df: pd.DataFrame) -> set:
    """
    Extract all items from the orders DataFrame.
    Args:
        df: pandas DataFrame with orders
    Returns:
        set: Set of all items
    """
    all_items = set()
    for items_str in df['items']:
        if pd.notna(items_str):
            # Split by semicolon and clean each item
            items = [item.strip() for item in items_str.split(';') if item.strip()]
            all_items.update(items)
    return all_items

def validate_items_consistency(orders_df: pd.DataFrame, items_df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate consistency between orders and items files.
    Args:
        orders_df: DataFrame with orders
        items_df: DataFrame with individual items
    Returns:
        tuple: (is_valid, list of inconsistencies)
    """
    # Get all items from orders file
    orders_items = extract_items_from_orders(orders_df)

    # Get all items from items file
    items_descriptions = set(items_df['description'].dropna())

    # Find inconsistencies
    inconsistencies = []

    # Check for items in orders but not in items file
    missing_in_items = orders_items - items_descriptions
    if missing_in_items:
        inconsistencies.append("Items found in orders but missing in items file:")
        for item in sorted(missing_in_items):
            inconsistencies.append(f"- {item}")

    # Check for items in items file but not in orders
    missing_in_orders = items_descriptions - orders_items
    if missing_in_orders:
        inconsistencies.append("\nItems found in items file but missing in orders:")
        for item in sorted(missing_in_orders):
            inconsistencies.append(f"- {item}")

    return len(inconsistencies) == 0, inconsistencies

def verify_file_types(orders_file: str, items_file: str) -> bool:
    """
    Verify that the files are of the correct type by checking their headers.
    Args:
        orders_file: Path to the supposed orders file
        items_file: Path to the supposed items file
    Returns:
        bool: True if files are of correct type
    """
    try:
        # Read first line (headers) of both files
        orders_df = pd.read_csv(orders_file, nrows=0)
        items_df = pd.read_csv(items_file, nrows=0)

        # Check for characteristic columns in orders file
        orders_columns = {'order id', 'items', 'date'}
        if not orders_columns.issubset(orders_df.columns):
            print(f"Error: '{orders_file}' does not appear to be an orders file (missing required columns: {orders_columns - set(orders_df.columns)})")
            return False

        # Check for characteristic columns in items file
        items_columns = {'order id', 'description', 'quantity'}
        if not items_columns.issubset(items_df.columns):
            print(f"Error: '{items_file}' does not appear to be an items file (missing required columns: {items_columns - set(items_df.columns)})")
            return False

        return True
    except Exception as e:
        print(f"Error verifying file types: {str(e)}")
        return False

def parse_euro_amount(amount_str: str) -> float:
    """
    Parse a Euro amount string to float.
    Args:
        amount_str: String representing an amount (e.g. "€10,99")
    Returns:
        float: The parsed amount
    """
    if pd.isna(amount_str):
        return 0.0
    # Remove € symbol and convert German format to standard float
    cleaned = amount_str.replace('€', '').replace(',', '.').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def validate_order_totals(orders_df: pd.DataFrame, items_df: pd.DataFrame) -> tuple[bool, list, dict]:
    """
    Validate that order totals match the sum of their item prices.
    Args:
        orders_df: DataFrame with orders
        items_df: DataFrame with individual items
    Returns:
        tuple: (is_valid, list of discrepancies, dict of order_id to price difference)
    """
    discrepancies = []
    total_orders = len(orders_df)
    discrepancy_count = 0
    mismatching_orders = []  # List of tuples (order_id, difference)
    price_differences = {}  # Dictionary to store all price differences

    # Process each order
    for _, order_row in orders_df.iterrows():
        order_id = order_row['order id']

        # Calculate order total including gift amount
        order_total = parse_euro_amount(order_row['total'])
        gift_amount = parse_euro_amount(order_row['gift'])
        total_with_gift = order_total + gift_amount

        # Get all items for this order
        order_items = items_df[items_df['order id'] == order_id]

        # Calculate total from items
        items_total = 0.0
        for _, item in order_items.iterrows():
            item_price = parse_euro_amount(item['price'])
            quantity = int(item['quantity']) if pd.notna(item['quantity']) else 1
            items_total += item_price * quantity

        # Compare totals (allowing for small floating point differences)
        difference = total_with_gift - items_total
        # Store all differences, even small ones
        price_differences[order_id] = difference

        if abs(difference) > 0.01:  # 1 cent tolerance
            discrepancy_count += 1
            mismatching_orders.append((order_id, difference))
            discrepancies.append(f"\nDiscrepancy found for order {order_id}:")
            discrepancies.append(f"Order total: €{total_with_gift:.2f} (base: €{order_total:.2f}, gift: €{gift_amount:.2f})")
            discrepancies.append(f"Sum of items: €{items_total:.2f}")
            discrepancies.append("Items in this order:")
            for _, item in order_items.iterrows():
                quantity = int(item['quantity']) if pd.notna(item['quantity']) else 1
                price = parse_euro_amount(item['price'])
                discrepancies.append(f"- {quantity}x {item['description']} at €{price:.2f} each")

    # Add summary at the end of discrepancies list
    if discrepancy_count > 0:
        discrepancies.append(f"\nPrice Validation Summary:")
        discrepancies.append(f"Total orders checked: {total_orders}")
        discrepancies.append(f"Orders with price discrepancies: {discrepancy_count}")
        discrepancies.append(f"Percentage of orders with discrepancies: {(discrepancy_count/total_orders*100):.1f}%")
        discrepancies.append(f"\nOrder IDs with price mismatches (positive difference means order total > sum of items):")
        # Sort by order ID and format the output
        for order_id, diff in sorted(mismatching_orders):
            discrepancies.append(f"{order_id}: €{diff:+.2f}")

    return discrepancy_count == 0, discrepancies, price_differences

def create_output_dataframe() -> pd.DataFrame:
    """
    Create a new DataFrame with the required output schema.
    Returns:
        pd.DataFrame: Empty DataFrame with the required columns
    """
    columns = [
        'Datum',          # Date
        'Wertstellung',   # Value date
        'Kategorie',      # Category
        'Name',           # Name
        'Verwendungszweck', # Purpose
        'Konto',          # Account
        'Bank',           # Bank
        'Betrag',         # Amount
        'Währung'         # Currency
    ]
    return pd.DataFrame(columns=columns)

def convert_date_format(date_str: str) -> str:
    """
    Convert date from YYYY-MM-DD to DD.MM.YYYY format.
    Args:
        date_str: Date string in YYYY-MM-DD format
    Returns:
        str: Date string in DD.MM.YYYY format
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%d.%m.%Y')
    except ValueError:
        return date_str

def format_verwendungszweck(order: pd.Series, header_line: str) -> str:
    """
    Format the Verwendungszweck field with the specified structure.
    Args:
        order: Series containing order data
        header_line: The variable header line to use
    Returns:
        str: Formatted Verwendungszweck text
    """
    parts = [header_line]

    # Always include reference and delivery address
    parts.append(f"Referenz: {order['order url']}")
    if pd.notna(order['to']) and order['to']:
        parts.append(f"Lieferadresse: {order['to']}")

    # Only include monetary fields if they have non-zero values
    shipping = parse_euro_amount(order['shipping'])
    if shipping != 0:
        parts.append(f"Versand: {order['shipping']}")

    shipping_refund = parse_euro_amount(order['shipping_refund'])
    if shipping_refund != 0:
        parts.append(f"Versanderstattung: {order['shipping_refund']}")

    gift = parse_euro_amount(order['gift'])
    if gift != 0:
        parts.append(f"Gutschein: {order['gift']}")

    refund = parse_euro_amount(order['refund'])
    if refund != 0:
        parts.append(f"Erstattung: {order['refund']}")

    # Only include difference if it's significant (more than 1 cent)
    difference = order['price_difference'] if pd.notna(order['price_difference']) else 0.0
    if abs(difference) > 0.01:
        # Format difference with 2 decimal places and German number format
        formatted_difference = str(difference).replace('.', ',')
        if ',' not in formatted_difference:
            formatted_difference += ',00'
        elif len(formatted_difference.split(',')[1]) == 1:
            formatted_difference += '0'
        parts.append(f"Differenz: {formatted_difference}")

    return ", ".join(parts)

def transform_step_1(orders_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    First transformation step: Create base output entries for each order.
    Args:
        orders_df: DataFrame containing the orders
        items_df: DataFrame with individual items
    Returns:
        pd.DataFrame: Output DataFrame with base entries
    """
    # Create empty output DataFrame
    output_df = create_output_dataframe()

    # Process each order
    output_rows = []
    for _, order in orders_df.iterrows():
        # Convert date to required format
        formatted_date = convert_date_format(order['date'])

        # Get total amount without currency symbol
        total_amount = parse_euro_amount(order['total'])
        if total_amount != 0:  # Only add if amount is not zero
            contra_row = {
                'Datum': formatted_date,
                'Wertstellung': formatted_date,
                'Name': order['order id'],
                'Kategorie': '',
                'Verwendungszweck': format_verwendungszweck(order, "Amazon Contra"),
                'Konto': '',
                'Bank': '',
                'Betrag': str(total_amount).replace('.', ','),
                'Währung': 'EUR'
            }
            output_rows.append(contra_row)

        # Get all items for this order
        order_items = items_df[items_df['order id'] == order['order id']]

        # Add a row for each item (considering quantity)
        for _, item in order_items.iterrows():
            # Get item price and quantity
            item_price = parse_euro_amount(item['price'])
            quantity = int(item['quantity']) if pd.notna(item['quantity']) else 1

            # Create a row for each unit of the item
            for _ in range(quantity):
                item_row = {
                    'Datum': formatted_date,
                    'Wertstellung': formatted_date,
                    'Name': order['order id'],
                    'Kategorie': '',
                    'Verwendungszweck': format_verwendungszweck(order, item['description']),
                    'Konto': '',
                    'Bank': '',
                    'Betrag': str(-item_price).replace('.', ','),  # Negative price
                    'Währung': 'EUR'
                }
                output_rows.append(item_row)

        # Add gift booking line if gift amount is greater than 0
        gift_amount = parse_euro_amount(order['gift'])
        if gift_amount > 0:
            gift_row = {
                'Datum': formatted_date,
                'Wertstellung': formatted_date,
                'Name': order['order id'],
                'Kategorie': '',
                'Verwendungszweck': format_verwendungszweck(order, "Gutscheinbuchung"),
                'Konto': '',
                'Bank': '',
                'Betrag': str(gift_amount).replace('.', ','),  # Positive gift amount
                'Währung': 'EUR'
            }
            output_rows.append(gift_row)

        # Add refund entries if refund amount is present
        refund_amount = parse_euro_amount(order['refund'])
        if refund_amount > 0:
            # Add refund row (positive amount)
            refund_row = {
                'Datum': formatted_date,
                'Wertstellung': formatted_date,
                'Name': order['order id'],
                'Kategorie': '',
                'Verwendungszweck': format_verwendungszweck(order, "Refund"),
                'Konto': '',
                'Bank': '',
                'Betrag': str(refund_amount).replace('.', ','),
                'Währung': 'EUR'
            }
            output_rows.append(refund_row)

            # Add contra refund row (negative amount)
            contra_refund_row = {
                'Datum': formatted_date,
                'Wertstellung': formatted_date,
                'Name': order['order id'],
                'Kategorie': '',
                'Verwendungszweck': format_verwendungszweck(order, "Contra refund"),
                'Konto': '',
                'Bank': '',
                'Betrag': str(-refund_amount).replace('.', ','),
                'Währung': 'EUR'
            }
            output_rows.append(contra_refund_row)

        # Add difference amount line
        difference = order['price_difference']
        if abs(difference) > 0.01:  # Only add if there's a significant difference
            diff_row = {
                'Datum': formatted_date,
                'Wertstellung': formatted_date,
                'Name': order['order id'],
                'Kategorie': '',
                'Verwendungszweck': format_verwendungszweck(order, "Differenzbetrag (Coupons etc.)"),
                'Konto': '',
                'Bank': '',
                'Betrag': str(-difference).replace('.', ','),  # Swap the sign of the difference
                'Währung': 'EUR'
            }
            output_rows.append(diff_row)

    # Add all rows to output DataFrame
    if output_rows:
        output_df = pd.concat([output_df, pd.DataFrame(output_rows)], ignore_index=True)

    return output_df

def process_csv(orders_file: str, items_file: str):
    """
    Process the input CSV files.
    Args:
        orders_file (str): Path to the orders CSV file
        items_file (str): Path to the items CSV file
    """
    try:
        # Check if both files exist
        if not Path(orders_file).is_file():
            print(f"Error: Orders file '{orders_file}' does not exist")
            sys.exit(1)
        if not Path(items_file).is_file():
            print(f"Error: Items file '{items_file}' does not exist")
            sys.exit(1)

        # Verify file types
        if not verify_file_types(orders_file, items_file):
            sys.exit(1)

        # Create work directory
        work_dir = create_work_directory(orders_file)

        # Read both CSV files
        orders_df = pd.read_csv(orders_file)
        items_df = pd.read_csv(items_file)

        # Check if last row of orders file is invalid and remove it if necessary
        original_row_count = len(orders_df)
        if len(orders_df) > 0 and not is_valid_order_row(orders_df.iloc[-1]):
            orders_df = orders_df.iloc[:-1]
            print(f"Removed last row from orders file as it did not contain valid order data")

        # Validate consistency between files
        is_valid, inconsistencies = validate_items_consistency(orders_df, items_df)
        if not is_valid:
            print("\nError: Inconsistencies found between orders and items files:")
            print('\n'.join(inconsistencies))
            sys.exit(1)

        # Validate order totals and store price differences
        is_valid, discrepancies, price_differences = validate_order_totals(orders_df, items_df)
        if not is_valid:
            print("\nWarning: Found discrepancies between order totals and item prices:")
            print('\n'.join(discrepancies))

        # Add price differences to orders DataFrame for later use
        orders_df['price_difference'] = orders_df['order id'].map(price_differences)

        # Save orders with price differences
        orders_with_diff_file = os.path.join(work_dir, 'orders_with_differences.csv')
        orders_df.to_csv(orders_with_diff_file, index=False)
        print(f"Orders with price differences have been saved to: {orders_with_diff_file}")

        # Apply first transformation step
        output_df = transform_step_1(orders_df, items_df)

        # Save output file
        output_file = os.path.join(work_dir, 'processed_output.csv')
        output_df.to_csv(output_file,
                        index=False,
                        sep=';',
                        encoding='utf-8')
        print(f"Output file has been created at: {output_file}")

        # Count total items
        total_items = orders_df['items'].apply(count_items).sum()

        # Prepare analytics output
        analytics = []
        analytics.append(f"CSV Files Analysis Report")
        analytics.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analytics.append(f"\nOrders File: {orders_file}")
        analytics.append(f"Items File: {items_file}")
        if original_row_count != len(orders_df):
            analytics.append(f"Original row count: {original_row_count:,}")
            analytics.append(f"Rows after removing invalid last row: {len(orders_df):,}")
        analytics.append(f"Number of rows in orders file: {len(orders_df):,}")
        analytics.append(f"Number of rows in items file: {len(items_df):,}")
        analytics.append(f"Number of columns in orders file: {len(orders_df.columns):,}")
        analytics.append(f"Number of columns in items file: {len(items_df.columns):,}")
        analytics.append(f"Total number of items: {total_items:,}")
        analytics.append("\nColumns in orders file:")
        for col in orders_df.columns:
            analytics.append(f"- {col}")
        analytics.append("\nColumns in items file:")
        for col in items_df.columns:
            analytics.append(f"- {col}")

        # Add formatted statistics for orders file
        analytics.append("\nDetailed Statistics (Orders File):")
        analytics.extend(format_statistics(orders_df))

        # Add items per row statistics
        items_per_row = orders_df['items'].apply(count_items)
        analytics.append("\nItems per Row Analysis:")
        analytics.append("-" * 40)
        analytics.append(f"Average items per row: {items_per_row.mean():.2f}")
        analytics.append(f"Maximum items in a row: {items_per_row.max():,}")
        analytics.append(f"Minimum items in a row: {items_per_row.min():,}")
        analytics.append(f"Rows with multiple items: {len(items_per_row[items_per_row > 1]):,}")
        analytics.append(f"Rows with single item: {len(items_per_row[items_per_row == 1]):,}")
        analytics.append(f"Rows with no items: {len(items_per_row[items_per_row == 0]):,}")

        # Remove duplicates and save deduplicated data
        df_deduped, duplicates_removed = remove_duplicates(orders_df)
        deduped_file = os.path.join(work_dir, 'data_deduped.csv')
        df_deduped.to_csv(deduped_file, index=False)

        # Add deduplication information to analytics
        analytics.append(f"\nDuplication Analysis:")
        analytics.append("-" * 40)
        analytics.append(f"Original row count: {len(orders_df):,}")
        analytics.append(f"Rows after deduplication: {len(df_deduped):,}")
        analytics.append(f"Duplicate rows removed: {duplicates_removed:,}")
        if duplicates_removed > 0:
            analytics.append(f"Duplication rate: {(duplicates_removed/len(orders_df)*100):.2f}%")

        # Write analytics to file
        metadata_file = os.path.join(work_dir, 'input_metadata')
        with open(metadata_file, 'w') as f:
            f.write('\n'.join(analytics))

        # Also print to console
        print('\n'.join(analytics))
        print(f"\nAnalytics have been saved to: {metadata_file}")
        print(f"Deduplicated data has been saved to: {deduped_file}")

    except Exception as e:
        print(f"Error processing the CSV files: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process Amazon order and item history CSV files")
    parser.add_argument("--orders", required=True, help="Path to the orders CSV file")
    parser.add_argument("--items", required=True, help="Path to the items CSV file")
    args = parser.parse_args()

    # Process the CSV files
    process_csv(args.orders, args.items)

if __name__ == "__main__":
    main()