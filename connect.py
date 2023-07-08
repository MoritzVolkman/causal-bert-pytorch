import csv

def extract_entries(input_file, output_file, num_entries):
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row
        j = 0
        entries = []
        for i, row in enumerate(reader):
            if j != 0 and not j % num_entries:
                break

            if len(row) >= 5 and not '\n' in row[2]:  # Check if the row has at least 5 columns
                tweet = str(row[2])
                retweet_count = row[4]
                entry = []
                entry.append(j)
                if isinstance(tweet, str):
                    entry.append(tweet)
                if retweet_count and int(float(retweet_count)) > 1.855:
                    entry.append('1')
                else:
                    entry.append('0')

                # Makes treatment 1 if biden is mentioned in tweet, 0 if trump is mentioned
                if '#biden' in tweet and retweet_count:
                    entry.append('1')
                    # makes confounder 1 if both are mentioned in tweet, else 0
                    if '#trump' in tweet:
                        entry.append('1')
                    else:
                        entry.append('0')
                elif '#trump' in tweet and not '#trump' in tweet and retweet_count:
                    entry.append('0')
                    if '#biden' in tweet:
                        entry.append('1')
                    else:
                        entry.append('0')
                if len(entry) == 5:
                    entries.append(entry)
                    j += 1
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        newheader = ['','tweet','retweet_count','treatment','confounder']
        writer.writerow(newheader)
        writer.writerows(entries)

    print(f'{j} entries extracted and saved to {output_file}.')


# Specify the input CSV files and the desired number of entries
file1 = 'hashtag_donaldtrump.csv'
file2 = 'hashtag_joebiden.csv'
output_file = 'combined.csv'
num_entries = 10000

# Extract entries from the first CSV file
extract_entries(file1, output_file, num_entries)

# Extract entries from the second CSV file and append them to the output file
extract_entries(file2, output_file, num_entries)
