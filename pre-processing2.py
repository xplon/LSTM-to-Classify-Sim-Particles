
#----------------------------------------------------------------------------------------------------------------------#
# Part of this code (and maybe some others in this project) is from codes
# written by my Professor Hualin Mei in this website:
#https://github.com/wlhwl/siMu_atm/tree/topic_1string_reco_BDT/detector_sim/ana_string/reco_one_string/scripts/notebooks
#----------------------------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import awkward as ak
import awkward_pandas
import uproot

df = pd.read_parquet("./data/train/mc_event.parquet")

merged_df = pd.DataFrame()

for string_ID in range(0, 124):
    path = f"./data/train/string_{string_ID}.root"
    file = uproot.open(path)
    # Access the tree
    tree = file["NoisyPmtHit"]

    pmthits = tree.arrays(library="pd")
    pmthits_ak = tree.arrays(library="ak")

    # signal = pmthits_ak.Type == 1

    pmthits_signal = ak.to_dataframe(pmthits_ak)

    unique_domId_counts = pmthits_signal.groupby('EventId')['DomId'].nunique()
    type(unique_domId_counts)
    # print(unique_domId_counts)

    # Filter the entries with at least 5 unique DomId
    filtered_event_ids = unique_domId_counts[unique_domId_counts >= 5].index
    filtered_df = pmthits_signal[pmthits_signal['EventId'].isin(filtered_event_ids)]

    # Create the unique_PMT_ID column
    filtered_df['unique_PMT_ID'] = filtered_df['DomId'] * 20 + filtered_df['PmtId']

    # print(filtered_df)

    # Step 1: Group by Dom_ID and get the minimal t0 value in each group
    min_t0_by_pmt = filtered_df.groupby(['EventId', 'DomId', 'Type'])['t0'].min().reset_index()

    min_t0_by_pmt['Type'] = (min_t0_by_pmt['Type'] + 1).astype('int') // 2
    filtered_df['Type'] = (filtered_df['Type'] + 1).astype('int') // 2

    # Step 2: Extract unique EventIds and DomIds
    unique_event_ids = min_t0_by_pmt['EventId'].unique()
    dom_ids = np.arange(21)  # np.arange(20)

    pecounts = filtered_df.groupby(['EventId', 'DomId', 'Type'])['t0'].count().reset_index()

    # Step 3: Create a new dataframe with zeros
    columns = [f'NDomId_{dom_id}' for dom_id in dom_ids]
    new_df = pd.DataFrame(0.0, index=unique_event_ids.astype('int'), columns=columns)

    # Step 4: Populate the new dataframe with t0 values
    for _, row in pecounts.iterrows():
        event_id = row['EventId'].astype('int')
        dom_id = row['DomId'].astype('int')
        t0_value = row['t0'].astype('int')
        if dom_id >= 21 * (string_ID + 1): continue
        # print (event_id, f'DomId_{dom_id}')
        new_df.at[event_id, f'NDomId_{dom_id - string_ID * 21}'] = t0_value

    new_column_names = {col: f'NDomId_{int(col.split("_")[1]) % 21}' for col in new_df.columns if
                        col.startswith('NDomId')}
    new_df = new_df.rename(columns=new_column_names)

    columns1 = [f'DomId_{dom_id}' for dom_id in dom_ids]
    new_df1 = pd.DataFrame(0.0, index=unique_event_ids.astype('int'), columns=columns1)

    for _, row in min_t0_by_pmt.iterrows():
        event_id = row['EventId'].astype('int')
        dom_id = row['DomId'].astype('int')
        t0_value = row['t0']
        if dom_id >= 21 * (string_ID + 1): continue
        # print (event_id, f'DomId_{dom_id}')
        new_df1.at[event_id, f'DomId_{dom_id - string_ID * 21}'] = t0_value

    new_column_names1 = {col: f'DomId_{int(col.split("_")[1]) % 21}' for col in new_df1.columns if
                         col.startswith('DomId')}
    new_df1 = new_df1.rename(columns=new_column_names1)

    new_df2 = filtered_df.groupby('EventId')['Type'].any().eq(1) + 0
    new_df2 = new_df2.reset_index()
    new_df2.rename(columns={'EventId': 'eventID'}, inplace=True)

    # new_df2 = new_df1.rename(columns={col: 'Type'})

    # Reset the index of new_df to make eventID a column and rename it to 'eventID'
    new_df = new_df.reset_index().rename(columns={'index': 'eventID'})

    new_df1 = new_df1.reset_index().rename(columns={'index': 'eventID'})

    # new_df2 = new_df2.reset_index().rename(columns={'index': 'eventID'})

    # Perform the merge operation
    merged_df1 = new_df2.merge(new_df1)
    merged_df1 = merged_df1.merge(new_df)
    merged_df1 = merged_df1.merge(df[['eventID', 'zenith_angle_out', 'azimuthal_angle_out']], on='eventID', how='left')

    # Renumber the eventID in order to convert all the data into a better form, in which wew can tell
    # which string did it from and which eventID it is originally.
    merged_df1['eventID'] += 1000000 * string_ID
    # Add all the outcomes to merged_df
    merged_df = pd.concat([merged_df, merged_df1], axis=0)

print(merged_df)

# This file will be processed into full-fixed_2.parquet
merged_df.to_parquet('./data/train/full.parquet')