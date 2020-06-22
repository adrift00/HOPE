import  pandas as pd
sparcc_data = pd.read_csv('/home/keyan/NewDisk/gut_data/SRR_data_new1/KO_predicted_filted.tsv', sep='\t')
cols = sparcc_data.columns.values.tolist()
rows = sparcc_data.index.values.tolist()
sparcc_mat = sparcc_data.to_numpy()
count = 0
for col in cols:
    if col == 'sequence':
        continue
    a=sparcc_data[col].sum()
    if a <= 20:
        sparcc_data.drop(col, axis=1, inplace=True)
    print(count, "finished")
    count = count +1
sparcc_data.to_csv('/home/keyan/NewDisk/gut_data/SRR_data_new1/KO_predicted_filted_1.tsv', sep='\t')
x = 1
