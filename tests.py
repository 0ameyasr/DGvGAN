# import pandas as pd

# # Load the source evaluation dataset
# df = pd.read_csv('evasion_evaluation_summary.csv')

# # Clean up and parse the TPR percentage to float values
# df['TPR_num'] = df['TPR (%)'].str.rstrip('%').astype('float')

# # Group by Technique, Rate, and Model to compute mean and std across seeds
# grouped = df.groupby(['Technique', 'Rate', 'Model'])['TPR_num'].agg(['mean', 'std']).reset_index()

# def format_row(row):
#     return f"{row['mean']:.2f}% ± {row['std']:.2f}%"

# grouped['TPR (%) mean +/- std'] = grouped.apply(format_row, axis=1)

# # Generate overall summary rows for each technique across all rates & models
# overall_rows = []
# for tech in df['Technique'].unique():
#     tech_df = df[df['Technique'] == tech]
#     m = tech_df['TPR_num'].mean()
#     s = tech_df['TPR_num'].std()
#     overall_rows.append({
#         'Technique': tech,
#         'Rate': 'All',
#         'Model': 'All',
#         'mean': m,
#         'std': s,
#         'TPR (%) mean +/- std': f"{m:.2f}% ± {s:.2f}%"
#     })
# overall_df = pd.DataFrame(overall_rows)

# final_rows = []
# for tech in df['Technique'].unique():
#     final_rows.append(grouped[grouped['Technique'] == tech])
#     final_rows.append(overall_df[overall_df['Technique'] == tech])

# final_df = pd.concat(final_rows, ignore_index=True)

# for technique in {"Benign Mimicry", "Stalling Padding", "Call Reordering"}:
#     print(final_df[final_df['Technique'] == technique], end='\n\n')
    
# final_df.to_csv('evasion_evaluation_summary_processed.csv', index=False)

### -------------------

# import json

# with open("vt_cache.json","r") as f:
#     data = json.loads(f.read())

# dates = []
# for entry in data:
#     entry_data = data[entry]
#     date = entry_data['First Seen on VirusTotal']
#     dates.append(int(date[:4]))

# print(f"Sample range: {min(dates)}-{max(dates)}")
# print(f"Samples w.r.t 2019: -{len(list(filter(lambda x: x <= 2019,dates)))} +{len(list(filter(lambda x: x > 2019,dates)))}")