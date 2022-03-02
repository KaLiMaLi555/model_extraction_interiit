import pandas as pd

def main():
    df_400 = pd.read_csv('kinetics400_test.csv')
    df_600 = pd.read_csv('kinetics600_test.csv')

    # Map label to a number
    map_400 = {}
    map_600 = {}
    
    # New empty dataframe
    df_400_new = pd.DataFrame(columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])
    df_600_new = pd.DataFrame(columns=['label', 'youtube_id', 'time_start', 'time_end', 'split'])

    # Iterate through the 400 test set
    for i in range(len(df_400)):
        # Get the values
        label = df_400.iloc[i]['label']
        youtube_id = df_400.iloc[i]['youtube_id']
        time_start = df_400.iloc[i]['time_start']
        time_end = df_400.iloc[i]['time_end']
        split = df_400.iloc[i]['split']
        # If the label is not in the map, add it
        if label not in map_400:
            map_400[label] = 1
            # Add the label to the new dataframe
            df_400_new = df_400_new.append({'label': label, 'youtube_id': youtube_id, 'time_start': time_start, 'time_end': time_end, 'split': split}, ignore_index=True)
        elif map_400[label] == 10:
            continue
        else:
            map_400[label] += 1
            # Add the label to the new dataframe
            df_400_new = df_400_new.append({'label': label, 'youtube_id': youtube_id, 'time_start': time_start, 'time_end': time_end, 'split': split}, ignore_index=True)

    # Iterate through the 600 test set
    for i in range(len(df_600)):
        # Get the values
        label = df_600.iloc[i]['label']
        youtube_id = df_600.iloc[i]['youtube_id']
        time_start = df_600.iloc[i]['time_start']
        time_end = df_600.iloc[i]['time_end']
        split = df_600.iloc[i]['split']
        # If the label is not in the map, add it
        if label not in map_600:
            map_600[label] = 1
            # Add the label to the new dataframe
            df_600_new = df_600_new.append({'label': label, 'youtube_id': youtube_id, 'time_start': time_start, 'time_end': time_end, 'split': split}, ignore_index=True)
        elif map_600[label] == 10:
            continue
        else:
            map_600[label] += 1
            # Add the label to the new dataframe
            df_600_new = df_600_new.append({'label': label, 'youtube_id': youtube_id, 'time_start': time_start, 'time_end': time_end, 'split': split}, ignore_index=True)

    # Save the dataframes
    df_400_new.to_csv('kinetics400_test_new.csv', index=False)
    df_600_new.to_csv('kinetics600_test_new.csv', index=False)

if __name__ == '__main__':
    main()