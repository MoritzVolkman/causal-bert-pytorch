import csv

def extract_entries(input_file, output_file, num_entries):
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row

        entries = []
        for i, row in enumerate(reader):
            if i >= num_entries:
                break

            if len(row) >= 5:  # Check if the row has at least 5 columns
                tweet = row[2]
                retweet_count = row[4]
                entry = []
                entry.append(tweet)
                if retweet_count and int(float(retweet_count)) > 99:
                    entry.append(1)
                else:
                    entry.append(0)

                # Makes treatment 1 if biden is mentioned in tweet, 0 if trump is mentioned
                if '#biden' in tweet and retweet_count:
                    entry.append(1)
                    # makes confounder 1 if both are mentioned in tweet, else 0
                    if '#trump' in tweet:
                        entry.append(1)
                    else:
                        entry.append(0)
                elif '#trump' in tweet and not '#trump' in tweet and retweet_count:
                    entry.append(0)
                    if '#biden' in tweet:
                        entry.append(1)
                    else:
                        entry.append(0)
                if len(entry) == 4:
                    entries.append(entry)

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        newheader = ['tweet','retweet_count','treatment','confounder']
        writer.writerow(newheader)
        writer.writerows(entries)

    print(f'{num_entries} entries extracted and saved to {output_file}.')


# Specify the input CSV files and the desired number of entries
file1 = 'hashtag_donaldtrump.csv'
file2 = 'hashtag_joebiden.csv'
output_file = 'combined.csv'
num_entries = 100000

# Extract entries from the first CSV file
extract_entries(file1, output_file, num_entries)

# Extract entries from the second CSV file and append them to the output file
extract_entries(file2, output_file, num_entries)