import csv
import os
import re
import time
from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests


def determine_winner(extracted_lines):
    if extracted_lines["win_or_tie"] == "|tie":
        return 0
    elif extracted_lines["win_or_tie"] == extracted_lines["player_p1"]:
        return 1
    elif extracted_lines["win_or_tie"] == extracted_lines["player_p2"]:
        return 2
    return None


def keep_before_comma(s):
    # Check if there's a comma in the string
    if ',' in s:
        # Split at the first comma and keep only the part before it
        return s.split(',', 1)[0]
    else:
        # If no comma, return the string as-is
        return s


def extract_lines_from_log(log_text):
    extracted_lines = {
        "player_p1": None,
        "player_p2": None,
        "switch_p1a": None,
        "switch_p2a": None,
        "win_or_tie": None
    }

    for line in log_text.splitlines():
        if line.startswith("|player|p1|") and not extracted_lines["player_p1"]:
            extracted_lines["player_p1"] = line.split('|')[3]
        elif line.startswith("|player|p2|") and not extracted_lines["player_p2"]:
            extracted_lines["player_p2"] = line.split('|')[3]
        elif line.startswith("|switch|p1a:") and not extracted_lines["switch_p1a"]:
            extracted_lines["switch_p1a"] = keep_before_comma(line.split('|')[3])
        elif line.startswith("|switch|p2a:") and not extracted_lines["switch_p2a"]:
            extracted_lines["switch_p2a"] = keep_before_comma(line.split('|')[3])
        elif line.startswith("|win|") and not extracted_lines["win_or_tie"]:
            extracted_lines["win_or_tie"] = line.split('|')[2].strip()

    if not extracted_lines["win_or_tie"]:
        extracted_lines["win_or_tie"] = "|tie"
    last_turn_line = next((line for line in reversed(log_text.splitlines()) if line.startswith('|turn|')), None)
    winner = determine_winner(extracted_lines)
    if last_turn_line:
        extracted_lines["turns"] = last_turn_line.split('|')[2].strip()
    else:
        extracted_lines["turns"] = 0
    extracted_lines["winner"] = winner

    return extracted_lines


class Lemons:
    def __init__(self):
        # initialize attributes here
        self.log = []
        self.duplicated_games = []
        self.missing_games = []
        self.all_rows = []
        self.confirmed_mess = []
        self.whitelist = []
        self.whitelist_filepath = "games_to_ignore.txt"
        self.fill_whitelist()
        self.rows_with_excess_players = []
        self.add_to_log("Program Start")

        pass

    def add_to_log(self, entry):
        # Get the current time in a readable format
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the time to the entry
        self.log.append(f"[{current_time}] {entry}")

    def fetch_and_save_log(self, link, download_dir="logs"):
        # Ensure download directory exists
        os.makedirs(download_dir, exist_ok=True)
        # Extract file name from link
        filename = link.replace("https://", "").replace("/", "_") + ".txt"
        file_path = os.path.join(download_dir, filename)

        # Check if log file has already been downloaded
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()

        # Download log if not already present
        if not link.startswith("http"):
            link = "https://" + link
        try:
            response = requests.get(link)
            response.raise_for_status()
            log_text = response.content.decode('utf-8')
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(log_text)
            return log_text
        except requests.RequestException as e:
            self.add_to_log(f"Failed to retrieve log from {link}: {e}")
            return None

    def process_csv(self, sub_input_csv, sub_output_csv, download_dir="logs"):
        link_cache = {}  # Cache to store already processed links
        rows_to_write = []  # Buffer for rows to be written in bulk

        with open(sub_input_csv, mode='r', newline='', encoding='utf-8', errors='replace') as csv_file, \
                open(sub_output_csv, mode='w', newline='', encoding='utf-8') as output_file:

            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                reader.fieldnames = []
            fieldnames = reader.fieldnames + ["player_p1", "player_p2", "switch_p1a", "switch_p2a", "winner", "turns"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            # Dictionary to fetch only rows where "winner" is empty, grouped by link
            links_to_fetch = defaultdict(list)

            # Reset reader to add rows to rows_to_write or links_to_fetch
            csv_file.seek(0)
            next(reader)  # Skip the header

            for row in reader:
                link = row.get("replay_link").replace("?p2", "")
                if link:

                    if row.get("winner"):  # If winner exists, add directly to rows_to_write
                        rows_to_write.append(row)
                    else:  # Group rows by link if winner is missing
                        links_to_fetch[link.strip()].append(row)

            with ThreadPoolExecutor(max_workers=10) as executor:
                # Schedule fetch tasks only for unique links that need to be fetched
                futures = {executor.submit(self.fetch_and_save_log, link, download_dir): link for link in
                           links_to_fetch}

                for future in as_completed(futures):
                    link = futures[future]
                    rows = links_to_fetch[link]  # List of rows with the same link
                    if link not in link_cache:  # Check cache for repeated links

                        log_text = future.result()
                        if log_text:
                            extracted_data = extract_lines_from_log(log_text)
                            link_cache[link] = extracted_data  # Cache this result
                        else:
                            extracted_data = {"player_p1": None, "player_p2": None, "switch_p1a": None,
                                              "switch_p2a": None,
                                              "winner": None, "turns": None}
                            link_cache[link] = extracted_data

                        for row in rows:  # Update all rows with the fetched data
                            row.update(link_cache[link])
                            row.pop("win_or_tie", None)
                            rows_to_write.append(row)  # Add each updated row to the buffer
                    else:
                        for row in rows:  # Update with cached data if already fetched
                            row.update(link_cache[link])
                            row.pop("win_or_tie", None)
                            rows_to_write.append(row)

            # Sort rows by 'number' column
            sorted_rows = sorted(rows_to_write, key=lambda r: int(r.get("number", 0)))

            # Write sorted rows to output
            writer.writerows(sorted_rows)

        # print(f"Total unique links processed: {len(link_cache)}")

    def download_xcl(self, sub_sheet_id, sub_sheet_name):
        url = f'https://docs.google.com/spreadsheets/d/{sub_sheet_id}/export?format=xlsx'

        # Download the Excel file
        response = requests.get(url)

        # Save to a file
        if response.status_code == 200:
            with open(sub_sheet_name, 'wb') as file:
                file.write(response.content)
            self.add_to_log("File downloaded successfully as 'downloaded_sheet.xlsx'")

        else:
            self.add_to_log(f"Failed to download file. Status code: {response.status_code}")

    def process_excel_file(self, file_path):
        # Load the Excel file
        excel_data = pd.ExcelFile(file_path)

        # Save the Importer sheet directly as a CSV
        importer_df = excel_data.parse("Importer")
        importer_df = importer_df[importer_df.iloc[:, 1].notna()]
        importer_df.columns = ['replay_num', 'replay_link']
        importer_df['replay_link'] = importer_df['replay_link'].str.replace(
            r"https?://smogtours\.psim\.us/battle-",
            "https://replay.pokemonshowdown.com/smogtours-",
            regex=True
        )
        importer_df.to_csv("importer.csv", index=False)

        # Initialize an empty list to hold each round's DataFrame
        rounds_data = []

        # Loop through each sheet name
        for sub_sheet_name in excel_data.sheet_names:
            if sub_sheet_name.startswith("Round"):
                # Parse the sheet
                df = excel_data.parse(sub_sheet_name)

                # Filter to keep only the columns named "Player" and "R"
                if 'Player' in df.columns and 'R' in df.columns:
                    # Rename columns
                    df = df.rename(columns={
                        df.columns[df.columns.str.contains("Player")][0]: "player_a",
                        df.columns[df.columns.str.contains("Player")][1]: "player_b",
                        "R": "r"
                    })

                    # Add sheet name as a new column
                    df["sheet_name"] = sub_sheet_name

                    # Append the modified DataFrame to the list
                    rounds_data.append(df[["player_a", "player_b", "r", "sheet_name"]])

        cleaned_rounds_df = self.clean_rounds(pd.concat(rounds_data, ignore_index=True))
        # Save the combined rounds data as a CSV
        cleaned_rounds_df.to_csv("rounds.csv", index=False)

    def clean_rounds(self, df):
        # Filter out rows where only one item exists in 'sheet_name'
        df = df[df['sheet_name'].str.len() > 1]

        # Initialize the winner and game_list columns
        df['winner'] = df['r'].str[0]  # First character goes to winner
        remainder = df['r'].str[1:]  # Remainder of the string after the first character

        # List to store indices of problematic rows
        problematic_rows = []

        def process_game_list(value, row_index):
            # Check if the value is NaN and handle it
            if pd.isna(value):
                return np.nan  # or return another default value if preferred

            # Strip non-numeric characters from remainder
            cleaned_value = re.sub(r'\D', '', value)

            # Check if cleaned remainder starts with a number
            if cleaned_value and cleaned_value[0].isdigit():
                # Check if length of digits is divisible by 4
                if len(cleaned_value) % 4 != 0:
                    # If not divisible by 4, log the row index and skip it
                    problematic_rows.append(row_index)
                    return np.nan  # Or any placeholder to indicate a problem with this row
                # Split into groups of 4 digits
                return ' '.join(re.findall(r'.{4}', cleaned_value))
            else:
                # If it doesn't start with a number, return "act"
                return "act"

        # Apply process_game_list to the remainder to create game_list
        df['game_list'] = [process_game_list(value, idx) for idx, value in remainder.items()]

        # Print all problematic rows at the end
        if problematic_rows:
            # print("The following rows had issues with divisibility into groups of 4 digits:")
            # print(df.loc[problematic_rows])
            subx = df.loc[problematic_rows]
            self.add_to_log(subx)

        # Filter out rows where both 'winner' and 'game_list' are null
        df = df[~df[['winner', 'game_list']].isnull().all(axis=1)]

        # Select the relevant columns to return
        return df[['player_a', 'player_b', 'winner', 'game_list', 'sheet_name']]

    def analyze_csv(self, file_path):
        all_numbers = []

        # Read the CSV and extract numbers
        with open(file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                game_list = row['game_list']
                self.all_rows.append(str(row))
                # Check if game_list is a list of numbers
                if game_list != "act":
                    # Split game_list by spaces to get individual numbers and add them to all_numbers
                    numbers = game_list.split()
                    all_numbers.extend(int(num) for num in numbers)

        # Use Counter to find duplicates
        number_counts = Counter(all_numbers)
        duplicates = sorted([num for num, count in number_counts.items() if count > 1], key=int)
        self.duplicated_games = duplicates
        if duplicates:
            self.add_to_log(f"Duplicates found in replay numbers: {duplicates}")
        else:
            self.add_to_log("No duplicates found in replay numbers")
        # Convert all_numbers to integers
        all_numbers = list(map(int, all_numbers))

        # Find the maximum number to establish the range
        max_number = max(all_numbers)

        # Find missing numbers from 1 to max_number
        full_range = set(range(1, max_number + 1))
        present_numbers = set(all_numbers)
        missing_numbers = sorted(full_range - present_numbers)
        missing_numbers = sorted(set(missing_numbers) - set(self.whitelist))
        self.missing_games = missing_numbers
        if missing_numbers:
            self.add_to_log(f"Sorted missing replay numbers: {missing_numbers}")
        else:
            self.add_to_log("No missing replay numbers")

    def fill_whitelist(self):
        with open(self.whitelist_filepath, 'r') as file:
            for line in file:
                # Remove leading/trailing whitespace, including spaces and tabs
                stripped_line = line.strip()

                # Skip lines that start with '#'
                if not stripped_line.startswith("#"):
                    # Remove any commas and split the line by whitespace
                    numbers = [int(num) for num in stripped_line.replace(',', '').split()]
                    # Extend the whitelist with these numbers
                    self.whitelist.extend(numbers)

    def check_num_players(self):

        rounds_df = pd.read_csv('rounds.csv')
        logs_df = pd.read_csv('processed_logs_optimized.csv')
        # Convert replay_num to a dictionary for quick lookup
        logs_dict = logs_df.set_index('replay_num')[['player_p1', 'player_p2']].to_dict('index')
        # Initialize a list to store num_players for each row in rounds_df
        num_players_list = []

        # Iterate over each row in rounds_df
        for _, row in rounds_df.iterrows():
            # Extract the game_list
            game_list = row['game_list']

            # Check if game_list contains 'act' and skip if it does
            if game_list.strip().lower() == 'act':
                num_players_list.append(0)
                continue

            # Split game_list into individual game IDs
            game_list = [int(game) for game in game_list.split()]

            # Check if any game in game_list is in the whitelist and skip the row if so
            if any(game in self.whitelist for game in game_list):
                num_players_list.append(0)  # or any placeholder value for skipped rows
                continue

            # Initialize a set to collect unique players
            players_set = set()

            # Loop through each game in game_list
            for game in game_list:
                # Get players for the game from logs_dict
                if game in logs_dict:
                    player_p1 = logs_dict[game]['player_p1']
                    player_p2 = logs_dict[game]['player_p2']
                    if isinstance(player_p1, str):
                        players_set.add(player_p1.lower())
                    if isinstance(player_p2, str):
                        players_set.add(player_p2.lower())

            # Append the number of unique players to num_players_list
            num_players_list.append(len(players_set))

        # Add num_players as a new column in rounds_df
        rounds_df['num_players'] = num_players_list

        # Save the updated DataFrame to a new CSV
        rows_with_more_than_two_players = rounds_df[rounds_df['num_players'] > 2]
        # Display rows with num_players > 2, if any
        if not rows_with_more_than_two_players.empty:
            self.add_to_log(f"Rows with more than 2 unique players:\n{rows_with_more_than_two_players}")
            self.rows_with_excess_players = rows_with_more_than_two_players
        else:
            self.add_to_log("No rows with num_players greater than 2.")

        # Save the updated DataFrame to a new CSV
        rounds_df.to_csv('rounds.csv', index=False)

    def check_dupes(self):
        df = pd.read_csv('processed_logs_optimized.csv')
        # Step 1: Find duplicate entries in the 'replay_link' column
        duplicates = df[df.duplicated(subset=['replay_link'], keep=False)]

        # Step 2: Find groups of duplicates where none of the 'replay_num' values are in the whitelist
        filtered_duplicates = duplicates.groupby('replay_link').filter(
            lambda group: not any(group['replay_num'].isin(lemon.whitelist))
        )

        # Check if there are any duplicates left after filtering
        if not filtered_duplicates.empty:
            log_message = "Grouped duplicates by 'replay_link' with associated 'replay_num' values (excluding " \
                          "whitelist):\n "

            # Group duplicates by 'replay_link' and collect 'replay_num' values
            grouped_duplicates = filtered_duplicates.groupby('replay_link')['replay_num'].apply(list)

            # Append each replay link and its list of replay numbers to the log message
            for replay_link, replay_nums in grouped_duplicates.items():
                log_message += f"Replay Link: {replay_link}\n"
                log_message += f"Associated Replay Numbers: {replay_nums}\n\n"

            # Add the final message to the log
            self.add_to_log(log_message)
        else:
            # No duplicates found message
            self.add_to_log("No duplicates found in 'replay_link' column or all duplicates are in the whitelist.")


if __name__ == "__main__":
    start_time = time.perf_counter()

    lemon = Lemons()

    # Call the function with the path to your Excel file
    # noinspection SpellCheckingInspection
    sheet_id = '12PyGiciXTqEj1ARWD-cM37l3mOoCgUsnUKqT5fpklgA'
    sheet_name = "current.xlsx"
    boo = True
    if boo:
        lemon.download_xcl(sheet_id, sheet_name)

    lemon.process_excel_file(sheet_name)
    lemon.analyze_csv("rounds.csv")

    if boo:
        input_csv = 'importer.csv'  # Input CSV with links under the header "link"
        output_csv = 'processed_logs_optimized.csv'  # Output CSV to store results
        lemon.add_to_log("Start link download")
        lemon.process_csv(input_csv, output_csv)
        lemon.add_to_log("End link download")

    lemon.check_dupes()

    # Load the CSVs
    lemon.check_num_players()

    for x in lemon.log:
        print(x)
        print("-" * 40)

    print(f"Execution time: {time.perf_counter() - start_time:.4f} seconds")  # End timing and print the result

    '''
888                                      
888                                      
888                                      
888 .d88b. 88888b.d88b.  .d88b. 88888b.  
888d8P  Y8b888 "888 "88bd88""88b888 "88b 
88888888888888  888  888888  888888  888 
888Y8b.    888  888  888Y88..88P888  888 
888 "Y8888 888  888  888 "Y88P" 888  888     

    '''
# TODO Add way to just pass dataframes around and only save/ access csvs once
# TODO stop passing file paths around. just put all that up in the init
# TODO make it so anytime something is passed to the log. if its empty there is a way to store that as well. So empty logs dont have to be joined with it.
# TODO Fix the Failed to retrieve log from https://match.conceeded.to.pkLeech: with some kind of whitelist or something. its annoying
