import csv

def calculate_retweet_count(csv_file, hashtag):
    total_retweet_count = 0

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        entries = 0
        for row in reader:
            tweet = row['tweet']
            if row['retweet_count']:
                retweet_count = int(float(row['retweet_count']))

                if hashtag in tweet.lower():
                    total_retweet_count += retweet_count
                    entries += 1

    return total_retweet_count, entries


# Specify the input CSV file
csv_file1 = 'hashtag_joebiden.csv'
csv_file2 = 'hashtag_donaldtrump.csv'

# Calculate the total retweet count for rows containing '#biden' in the 'tweet' column
total_count1 = calculate_retweet_count(csv_file1, '#biden')
total_count2 = calculate_retweet_count(csv_file2, '#trump')

print(f'Total retweet count for rows with "#biden": {total_count1} \n')
print(f'Total retweet count for rows with "#trump": {total_count2}')

average_retweet = (total_count1[0] + total_count2[0]) / (total_count1[1] + total_count2[1])
average_retweet_biden = total_count1[0] / total_count1[1]
average_retweet_trump = total_count2[0] / total_count2[1]

print(average_retweet, average_retweet_biden, average_retweet_trump)
