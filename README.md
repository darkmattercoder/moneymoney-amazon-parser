# MoneyMoney Amazon Parser

A Python script to process Amazon order history CSV files and convert them into a format suitable for importing into MoneyMoney.
Since the moneymoney amazon plugin [https://github.com/Michael-Beutling/Amazon-MoneyMoney](https://github.com/Michael-Beutling/Amazon-MoneyMoney) is not properly maintained anymore, this script is a workaround to import amazon orders into MoneyMoney.

It relies on the output format of the amazon order history reporter chrome extension by Philip Mulcahy [https://github.com/azad-codes/amazon-order-history-reporter](https://github.com/azad-codes/amazon-order-history-reporter).

In order to work, you need to create both order and item csv exports of the exact same time period with the chrome extension.

This script is a quick and dirty approach and does not handle all edge cases. It has been created with the extensive help of AI. And has not been further polished in terms of code quality. However, I needed a quick solution to get my amazon orders into MoneyMoney again. And this tool does the job.

The way how the script handles the different booking entries might not fit your needs. I basically tried to recreate the behavior of the *Mix*-account type that is present in the former MoneyMoney amazon plugin, where all orders consist of individual item bookings, contra bookings to match with other accounts as well as difference booking entries from gift cards and promotional discounts to match the total order amount.

## Features

- Processes Amazon order history and item history CSV files
- Validates data consistency between order and item files
- Detects and reports price discrepancies
- Handles gift cards, refunds, and shipping costs
- Generates detailed analytics about the processed data
- Creates a properly formatted output file for MoneyMoney import

## Requirements

- Python 3.6 or higher
- pandas library

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with two required arguments:
```bash
python process_csv.py --orders path/to/amazon_order_history.csv --items path/to/amazon_item_history.csv
```

The script will:
1. Create a timestamped work directory
2. Validate input files
3. Process the data
4. Generate several output files:
   - `processed_output.csv`: The main output file for MoneyMoney
   - `orders_with_differences.csv`: Orders with their price differences
   - `data_deduped.csv`: Deduplicated order data
   - `input_metadata`: Detailed analysis of the processed data

## Output Format

The generated `processed_output.csv` contains the following columns:
- Datum (Date)
- Wertstellung (Value date)
- Kategorie (Category)
- Name
- Verwendungszweck (Purpose)
- Konto (Account)
- Bank
- Betrag (Amount)
- Währung (Currency)

For each order, multiple rows are generated:
1. Main order entry (Amazon Contra)
2. Individual item entries
3. Gift card entry (if applicable)
4. Difference amount entry (if discrepancies exist)

## License

This project is licensed under the MIT License - see the LICENSE file for details.