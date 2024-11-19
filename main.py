import csv
import json
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


def load_aliases(json_path):
    """Load aliases from a JSON file."""
    with open(json_path, 'r') as f:
        alias_mapping = json.load(f)
    return alias_mapping


def normalize_name(name, alias_mapping):
    """Normalize a name to its canonical form using alias mapping."""
    normalized_name = name.strip().lower()  # Normalize input name
    for canonical, aliases in alias_mapping.items():
        if normalized_name in [alias.lower() for alias in aliases]:
            return canonical
    return name  # Return the original name if no alias matches


def count_mismatched_winners(csv_path='processed_rounds.csv'):
    try:
        # Load the processed rounds CSV file
        df = pd.read_csv(csv_path)

        # Normalize the columns by stripping whitespace and converting to lowercase
        df['winner_from_importer'] = df['winner_from_importer'].str.strip().str.lower()
        df['winner_from_rounds'] = df['winner_from_rounds'].str.strip().str.lower()

        # Count rows where the two columns do not match
        mismatched_count = (df['winner_from_importer'] != df['winner_from_rounds']).sum()

        return mismatched_count

    except FileNotFoundError:
        return "CSV file not found."
    except KeyError as e:
        return f"Missing column in the CSV: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


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
    def __init__(self, doc_id, importer_path="importer.csv", processed_importer_path="processed_logs.csv",
                 rounds_path="rounds.csv", allow_path="allow.txt", xcl_path="current.xlsx", logs_direct="logs",
                 processed_rounds_path="processed_rounds.csv", alias_path="merged_file.json",
                 mismatched_path="mismatch.csv"):
        # initialize attributes here
        self.log = []
        self.sheet_id = doc_id
        self.duplicated_games = []
        self.missing_games = []
        self.all_rows = []
        self.confirmed_mess = []
        self.allow_list = []
        self.allow_list_filepath = allow_path
        self.fill_allow_list()
        self.rows_with_excess_players = []
        self.importer_sheet_path = importer_path
        self.processed_importer_sheet_path = processed_importer_path
        self.rounds_sheet_path = rounds_path
        self.xcl_file_path = xcl_path
        self.logs_directory = logs_direct
        self.processed_rounds_sheet_path = processed_rounds_path
        self.alias_file_path = alias_path
        self.mismatched_winners_file_path = mismatched_path
        # self.add_to_log("Program Start")

        pass

    def add_to_log(self, entry):
        # Get the current time in a readable format
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the time to the entry
        self.log.append(f"[{current_time}] {entry}")

    def fetch_and_save_log(self, link):
        # Ensure download directory exists
        os.makedirs(self.logs_directory, exist_ok=True)
        # Extract file name from link
        filename = link.replace("https://", "").replace("/", "_") + ".txt"
        file_path = os.path.join(self.logs_directory, filename)

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
            self.add_to_log(f"Failed to retrieve log from {link}")
            return None

    def process_csv(self):
        link_cache = {}  # Cache to store already processed links
        rows_to_write = []  # Buffer for rows to be written in bulk

        with open(self.importer_sheet_path, mode='r', newline='', encoding='utf-8', errors='replace') as csv_file, \
                open(self.processed_importer_sheet_path, mode='w', newline='', encoding='utf-8') as output_file:

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
                futures = {executor.submit(self.fetch_and_save_log, link): link for link in
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

    def download_xcl(self):
        url = f'https://docs.google.com/spreadsheets/d/{lemon.sheet_id}/export?format=xlsx'

        # Download the Excel file
        response = requests.get(url)

        # Save to a file
        if response.status_code == 200:
            with open(lemon.xcl_file_path, 'wb') as file:
                file.write(response.content)
            self.add_to_log("File downloaded successfully as 'downloaded_sheet.xlsx'")

        else:
            self.add_to_log(f"Failed to download file. Status code: {response.status_code}")

    def process_excel_file(self):
        # Load the Excel file
        excel_data = pd.ExcelFile(self.xcl_file_path)

        # Save the Importer sheet directly as a CSV
        importer_df = excel_data.parse("Importer")
        importer_df = importer_df[importer_df.iloc[:, 1].notna()]
        importer_df.columns = ['replay_num', 'replay_link']
        importer_df['replay_link'] = importer_df['replay_link'].str.replace(
            r"https?://smogtours\.psim\.us/battle-",
            "https://replay.pokemonshowdown.com/smogtours-",
            regex=True
        )
        importer_df.to_csv(self.importer_sheet_path, index=False)

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
        cleaned_rounds_df.to_csv(self.rounds_sheet_path, index=False)

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
            self.add_to_log(f"Rows with r columns with messed up digits\n {subx}")

        # Filter out rows where both 'winner' and 'game_list' are null
        df = df[~df[['winner', 'game_list']].isnull().all(axis=1)]

        # Select the relevant columns to return
        return df[['player_a', 'player_b', 'winner', 'game_list', 'sheet_name']]

    def analyze_csv(self):
        all_numbers = []

        # Read the CSV and extract numbers
        with open(self.rounds_sheet_path, mode='r') as file:
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
            # self.add_to_log("No duplicates found in replay numbers")
            pass
        # Convert all_numbers to integers
        all_numbers = list(map(int, all_numbers))

        # Find the maximum number to establish the range
        max_number = max(all_numbers)

        # Find missing numbers from 1 to max_number
        full_range = set(range(1, max_number + 1))
        present_numbers = set(all_numbers)
        missing_numbers = sorted(full_range - present_numbers)
        missing_numbers = sorted(set(missing_numbers) - set(self.allow_list))
        self.missing_games = missing_numbers
        if missing_numbers:
            self.add_to_log(f"Sorted missing replay numbers: {missing_numbers}")
        else:
            self.add_to_log("No missing replay numbers")

    def fill_allow_list(self):
        with open(self.allow_list_filepath, 'r') as file:
            for line in file:
                # Remove leading/trailing whitespace, including spaces and tabs
                stripped_line = line.strip()

                # Skip lines that start with '#'
                if not stripped_line.startswith("#"):
                    # Remove any commas and split the line by whitespace
                    numbers = [int(num) for num in stripped_line.replace(',', '').split()]
                    # Extend the allow_list with these numbers
                    self.allow_list.extend(numbers)

    def check_num_players(self):

        rounds_df = pd.read_csv(self.rounds_sheet_path)
        logs_df = pd.read_csv(self.processed_importer_sheet_path)
        # Convert replay_num to a dictionary for quick lookup
        logs_dict = logs_df.set_index('replay_num')[['player_p1', 'player_p2']].to_dict('index')
        # Initialize a list to store num_players for each row in rounds_df
        num_players_list = []
        temp_log = ""
        # Iterate over each row in rounds_df
        for _, row in rounds_df.iterrows():
            # Extract the game_list
            game_list = row['game_list']

            # Check if game_list contains 'act' and skip if it does
            try:
                # Check if game_list contains 'act' and skip if it does
                if game_list.strip().lower() == 'act':
                    num_players_list.append(0)
                    continue
            except AttributeError:

                temp_log += (
                    f"\nAttributeError: game_list value is {game_list}, which is of type {type(game_list)} \n"
                    f"\tProblematic row: player_a: {row['player_a']}, player_b: {row['player_b']}, sheet_name: {row['sheet_name']}")
                num_players_list.append(99)
                continue

            # Split game_list into individual game IDs
            game_list = [int(game) for game in game_list.split()]

            # Check if any game in game_list is in the allow_list and skip the row if so
            if any(game in self.allow_list for game in game_list):
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
        if not temp_log == "":
            self.add_to_log(temp_log)
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
        rounds_df.to_csv(self.rounds_sheet_path, index=False)

    def check_dupes(self):
        df = pd.read_csv(self.processed_importer_sheet_path)
        # Step 1: Find duplicate entries in the 'replay_link' column
        duplicates = df[df.duplicated(subset=['replay_link'], keep=False)]

        # Step 2: Find groups of duplicates where none of the 'replay_num' values are in the allow_list
        filtered_duplicates = duplicates.groupby('replay_link').filter(
            lambda group: not any(group['replay_num'].isin(lemon.allow_list))
        )

        # Check if there are any duplicates left after filtering
        if not filtered_duplicates.empty:
            log_message = "Grouped duplicates by 'replay_link' with associated 'replay_num' values (excluding " \
                          "allow_list):\n "

            # Group duplicates by 'replay_link' and collect 'replay_num' values
            grouped_duplicates = filtered_duplicates.groupby('replay_link')['replay_num'].apply(list)

            # Append each replay link and its list of replay numbers to the log message
            for replay_link, replay_nums in grouped_duplicates.items():
                log_message += f"Replay Link: {replay_link}\n"
                log_message += f"Associated Replay Numbers: {replay_nums}\n"

            # Add the final message to the log
            self.add_to_log(log_message)
        else:
            # No duplicates found message
            # self.add_to_log("No duplicates found in 'replay_link' column or all duplicates are in the allow_list.")
            pass

    def get_winner_from_replay(self, replay_num):
        try:
            csv_path = self.processed_importer_sheet_path
            # Load the CSV file
            df = pd.read_csv(csv_path)

            # Find the row with the specified replay_num
            row = df[df['replay_num'] == replay_num]

            # If the replay_num is not found
            if row.empty:
                return "Replay number not found in the CSV."

            # Extract winner and player information
            winner = row.iloc[0]['winner']
            player_p1 = row.iloc[0]['player_p1']
            player_p2 = row.iloc[0]['player_p2']

            # Determine the winner
            if winner == 1:
                return player_p1
            elif winner == 2:
                return player_p2
            else:
                return "Problem in assigning winner."

        except FileNotFoundError:
            return "CSV file not found."
        except KeyError as e:
            return f"Missing column in the CSV: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

    def process_rounds_and_determine_winners(self):
        try:
            csv_path = self.rounds_sheet_path
            # Load the rounds CSV file
            df = pd.read_csv(self.rounds_sheet_path)

            # Get the first non-header row
            first_row = df.iloc[0]

            # Extract the game_list and split into game numbers
            game_list = first_row['game_list']
            game_numbers = map(int, game_list.split())

            # Collect winners from games using get_winner_from_replay
            winners = []
            for game in game_numbers:
                winner = lemon.get_winner_from_replay(game)
                winners.append(winner)

            # Determine the most common winner
            winner_from_importer = Counter(winners).most_common(1)[0][0]

            # Determine the winner_from_rounds
            winner_col = first_row['winner']
            if winner_col == 'a':
                winner_from_rounds = first_row['player_a']
            else:
                winner_from_rounds = first_row['player_b']

            return winner_from_importer, winner_from_rounds

        except FileNotFoundError:
            return "CSV file not found."
        except KeyError as e:
            return f"Missing column in the CSV: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

    def process_all_rounds_and_add_winners(self):
        try:
            # Load the rounds CSV file
            df = pd.read_csv(self.rounds_sheet_path)

            # Initialize new columns
            df['winner_from_importer'] = None
            df['winner_from_rounds'] = None

            # Iterate over each row in the DataFrame
            for index, row in df.iterrows():
                # Extract the game_list and split into game numbers
                game_list = row['game_list']
                if game_list.startswith("act"):
                    df.at[index, 'winner_from_importer'] = "act"
                    df.at[index, 'winner_from_rounds'] = "act"
                    continue  # Skip further processing for this row
                game_numbers = map(int, game_list.split())

                # Collect winners from games using get_winner_from_replay
                winners = []
                for game in game_numbers:
                    winner = self.get_winner_from_replay(game)
                    winners.append(winner)

                # Determine the most common winner for this round
                winner_from_importer = Counter(winners).most_common(1)[0][0] if winners else "No winners found"

                # Determine the winner_from_rounds
                winner_col = row['winner']
                if winner_col.lower() == 'a':
                    winner_from_rounds = row['player_a']
                elif winner_col.lower() == 'b':
                    winner_from_rounds = row['player_b']
                else:
                    winner_from_rounds = "Invalid winner column value"

                # Update the DataFrame with the new values
                df.at[index, 'winner_from_importer'] = winner_from_importer
                df.at[index, 'winner_from_rounds'] = winner_from_rounds

            # Save the updated DataFrame back to the CSV
            df.to_csv(self.processed_rounds_sheet_path, index=False)

            print(f"Updated {self.processed_rounds_sheet_path} with winner_from_importer and winner_from_rounds.")

        except FileNotFoundError:
            print("CSV file not found.")
        except KeyError as e:
            print(f"Missing column in the CSV: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def count_mismatched_winners_with_aliases(self):
        try:
            # Load the processed rounds CSV file
            df = pd.read_csv(self.processed_rounds_sheet_path)

            # Load alias mapping from JSON
            alias_mapping = load_aliases(self.alias_file_path)

            # Normalize columns by resolving aliases
            df['winner_from_importer'] = df['winner_from_importer'].apply(lambda x: normalize_name(x, alias_mapping))
            df['winner_from_rounds'] = df['winner_from_rounds'].apply(lambda x: normalize_name(x, alias_mapping))

            # Count rows where the two columns do not match
            mismatched_count = (df['winner_from_importer'] != df['winner_from_rounds']).sum()

            return mismatched_count

        except FileNotFoundError:
            return "CSV file not found."
        except KeyError as e:
            return f"Missing column in the CSV: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

    def save_mismatched_winners_to_csv(self):
        try:
            csv_path = self.processed_rounds_sheet_path
            alias_json = self.alias_file_path
            output_path = self.mismatched_winners_file_path
            # Load the processed rounds CSV file
            df = pd.read_csv(csv_path)

            for column in df.select_dtypes(include=['object']):  # Select columns with string data types
                df[column] = df[column].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

            # Load alias mapping from JSON
            alias_mapping = load_aliases(alias_json)

            # Normalize columns by resolving aliases
            df['normalized_winner_from_importer'] = df['winner_from_importer'].apply(
                lambda x: normalize_name(x, alias_mapping))
            df['normalized_winner_from_rounds'] = df['winner_from_rounds'].apply(
                lambda x: normalize_name(x, alias_mapping))

            # Filter out the rows where the winners don't match
            mismatched_rows = df[df['normalized_winner_from_importer'] != df['normalized_winner_from_rounds']]

            # If there are mismatched rows, save them to a CSV
            # Check if mismatched rows exist
            if not mismatched_rows.empty:
                # Group by 'normalized_winner_from_rounds' and sort by the size of each group
                grouped = mismatched_rows.groupby('normalized_winner_from_rounds').size().reset_index(name='group_size')
                grouped = grouped.sort_values(by='group_size', ascending=False)

                # Sort the original DataFrame by the group sizes and 'normalized_winner_from_rounds'
                sorted_rows = mismatched_rows.set_index('normalized_winner_from_rounds').loc[
                    grouped['normalized_winner_from_rounds']].reset_index()

                # Save the sorted mismatched rows to CSV
                sorted_rows.to_csv(output_path, index=False)
                print(f"Saved sorted mismatched rows to {output_path}")
            else:
                print("No mismatched rows found.")

        except FileNotFoundError:
            print("CSV file not found.")
        except KeyError as e:
            print(f"Missing column in the CSV: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_grouped_winners_to_json(self):
        mismatch_csv = "mismatch.csv"
        output_json = 'grouped_winners.json'
        try:
            # Load the mismatched winners CSV file
            df = pd.read_csv(mismatch_csv)

            # Group by both 'normalized_winner_from_rounds' and 'normalized_winner_from_importer'
            grouped = df.groupby(
                ['normalized_winner_from_rounds', 'normalized_winner_from_importer']).size().reset_index(
                name='group_size')

            # Filter out the groups that have exactly 2 entries
            exact_two_groups = grouped[grouped['group_size'] == 2]

            # Dictionary to store unique winners by normalized winner from rounds
            winners_dict = {}

            # For each group, find all the rows in that group
            for _, row in exact_two_groups.iterrows():
                winner_from_round = row['normalized_winner_from_rounds'].lower()
                winner_from_importer = row['normalized_winner_from_importer'].lower()

                # Add to the dictionary if not already present
                if winner_from_round not in winners_dict:
                    winners_dict[winner_from_round] = {}

                if winner_from_importer not in winners_dict[winner_from_round]:
                    winners_dict[winner_from_round][winner_from_importer] = 0

                winners_dict[winner_from_round][winner_from_importer] += 1

            # Convert dictionary to a structure suitable for JSON output
            for round_key in winners_dict:
                winners_dict[round_key] = list(winners_dict[round_key].keys())

            # Save the result to a JSON file
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(winners_dict, f, indent=4)

            print(f"Grouped winners saved to {output_json}")

        except FileNotFoundError:
            print("Mismatched winners CSV file not found.")
        except KeyError as e:
            print(f"Missing column in the CSV: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_time = time.perf_counter()

    # Call the function with the path to your Excel file
    # noinspection SpellCheckingInspection
    # basic_test is a copy of ADVR2 doc with errors I put in myself. but only a few
    # temp_test is a copy of the doc from the past. lots of errors
    # real_sheet_id should be self explanatory
    basic_test = '1VAFXclvNu1edSAI0XbGX_iGAEehXNDWLA9xgnFvGLbM'
    temp_test = '1Pbb5gvvV3u2ckk4LDI6sYzb7fPXqd_9M-_6tNzobsv4'
    real_sheet_id = '12PyGiciXTqEj1ARWD-cM37l3mOoCgUsnUKqT5fpklgA'
    temp_temp_test = '1qhS1QSBnPoCG6S5wcXV-TzoagRqmHyjqTctIiBTDkFM'
    test = False
    sheet_id = temp_test if test else real_sheet_id
    # sheet_id = basic_test
    # sheet_id = temp_temp_test
    lemon = Lemons(doc_id=sheet_id)
    lemon.add_to_log(f"Test: {test} Sheet ID: {lemon.sheet_id}")
    boo = True
    if boo:
        lemon.download_xcl()

    lemon.process_excel_file()
    lemon.analyze_csv()

    if boo:
        # lemon.add_to_log("Start link download")
        lemon.process_csv()
        # lemon.add_to_log("End link download")
    lemon.check_dupes()
    # Load the CSVs
    lemon.check_num_players()
    for x in lemon.log:
        print(x)
        print("-" * 40)

    # x = lemon.get_winner_from_replay(7)
    # print(x)

    lemon.process_all_rounds_and_add_winners()

    mismatches = count_mismatched_winners()
    print("Number of mismatched rows:", mismatches)

    mismatches = lemon.count_mismatched_winners_with_aliases()
    print("Number of mismatched rows:", mismatches)

    lemon.save_mismatched_winners_to_csv()
    print(f"Execution time: {time.perf_counter() - start_time:.4f} seconds")  # End timing and print the result
    lemon.save_grouped_winners_to_json()

    # Initialize an empty dictionary
    grouped_winners = {}

    # Read the CSV file
    with open('grouped_winners.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            round_number = row['rounds']
            importer = row['importer']

            # If the round number already exists in the dictionary, append the importer
            if round_number in grouped_winners:
                grouped_winners[round_number].append(importer)
            else:
                # Otherwise, create a new entry with a list containing the importer
                grouped_winners[round_number] = [importer]

    # Save the dictionary to a file called temp_json
    with open('temp_json.json', 'w') as json_file:
        json.dump(grouped_winners, json_file, indent=4)

    print("JSON data saved to 'temp_json.json'")

    with open('mismatch.csv', mode='r') as infile:
        csv_reader = csv.DictReader(infile)

        # Define the fieldnames (columns you want to keep)
        fieldnames = ['winner_from_rounds', 'winner_from_importer', 'game_list', "other_player", "sheet_name"]

        # Open the reduced_mismatch.csv file in write mode
        with open('reduced_mismatch.csv', mode='w', newline='') as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            # Write the header
            csv_writer.writeheader()
            other_player = ""
            # Iterate over the rows of the original file and write the reduced rows
            for row in csv_reader:
                other_player = row['player_b'] if row['player_a'] == row['normalized_winner_from_rounds'] else row[
                    'player_a']
                reduced_row = {
                    'winner_from_rounds': row['normalized_winner_from_rounds'],
                    'winner_from_importer': row['normalized_winner_from_importer'],
                    'game_list': row['game_list'],
                    'other_player': other_player,
                    'sheet_name': row['sheet_name']

                }
                csv_writer.writerow(reduced_row)

    print("Reduced CSV saved to 'reduced_mismatch.csv'")

    '''with open('reduced_mismatch.csv', mode='r') as infile:
        csv_reader = csv.DictReader(infile)

        # Create a list to store the rows where winner_from_importer == other_player
        matching_rows = []

        # Iterate over each row in the CSV
        for row in csv_reader:
            if row['winner_from_importer'] == row['other_player']:
                matching_rows.append(row)

    # Print the matching rows or save them to a new CSV if needed
    if matching_rows:
        print("Matching rows found:")
        for row in matching_rows:
            print(row)

        # Optionally, write the matching rows to a new CSV
        with open('matching_rows.csv', mode='w', newline='') as outfile:
            fieldnames = csv_reader.fieldnames  # Use the original headers
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            # Write the header and then the matching rows
            csv_writer.writeheader()
            csv_writer.writerows(matching_rows)

        print("Matching rows saved to 'matching_rows.csv'")
    else:
        print("No matching rows found.")'''

    '''import json

    # Load the JSON data from both files
    with open('temp_json.json', 'r') as f1:
        data1 = json.load(f1)

    with open('alias.json', 'r') as f2:
        data2 = json.load(f2)

    # Merge the dictionaries
    merged_data = data1.copy()  # Start with data1

    # Loop through the second dictionary and merge
    for key, value in data2.items():
        if key in merged_data:
            # If the key exists in both, merge the lists
            merged_data[key].extend(value)  # Add values from data2 to data1
        else:
            # If the key only exists in data2, add it
            merged_data[key] = value

    # Save the merged data back to a JSON file
    with open('merged_file.json', 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)

    print("Files merged successfully!")'''

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
#  TODO make it so anytime something is passed to the log. if its empty there is a
#   way to store that as well. So empty logs dont have to be joined with it.
#  TODO Fix the Failed to retrieve log from
#   https://match.conceeded.to.pkLeech: with some kind of allow_list or something. its annoying

# TODO stop the blank html file being downloaded in logs called http. I think its from blank line at end of importer?
# TODO have it check to see if there are any more games or whatever and report how many games have been added
# TODO have anything that can add a log be its own method/function. To make everything more compact and easier
