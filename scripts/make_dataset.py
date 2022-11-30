#!/usr/bin/env python
# coding: utf-8

def main():
    # In[ ]:


    from pathlib import Path
    from dask.distributed import Client
    from dask_cuda import LocalCUDACluster
    import dask_cudf
    import cudf
    import numpy as np
    import logging
    import os


    # In[ ]:


    input_filepath = '/mnt/DATA/nfs/data/new_marref/MarRef.speciestrain.fasta'
    output_folder_path = '/mnt/DATA/nfs/data/gene-cnn'
    label_filepath = '/mnt/DATA/nfs/data/new_marref/MarRef.taxlabel.tsv'


    # In[ ]:


    kmer = 1
    shuffle_data = True
    perc_of_labs = 1
    perc_of_data = .1
    data_splits = {
        'train': 0.8,
        'val': 0.1,
        'test': 0.1,
    }


    dataset_name = 'dset-10pD-100pL'
    split_mers_to_cols = True
    raw_cols = ['seq', 'id']
    raw_seq_col = 'seq'
    raw_lab_col = 'id'

    random_seed = 42
    do_random_seed = True


    label_file_id_col = 'id'
    label_file_label_cols = [
        { 'name_col': 'Species', 'code_col': 'species_code'},
        { 'name_col': 'Genus', 'code_col': 'genus_code'},
    ]

    possible_gene_values = ['A', 'C', 'G', 'T']  
    max_input_len = 150


    # In[ ]:


    def validate_label_file_label_cols(label_file_label_cols):
        for ith in label_file_label_cols:
            type_list = [type(val) for key, val in ith.items()]
            is_string = sum([x == str for x in type_list])
            assert is_string > 0, f"\nat least 1 of 'name_col' or 'code_col' must be a string, in 'label_file_label_cols'. \ncurrently, set as {ith}"
            only_acceptable_types = all([t == type(None) if t != str else True for t in type_list ])
            assert only_acceptable_types, f"\nthe only acceptable values for 'name_col' or 'code_col' are str type and None type. \ncurrently, set as {ith}"
        print("values of label_file_label_cols are acceptable")

    def validate_perc_of_labs(perc_of_labs):
        if type(perc_of_labs) == int:
            perc_of_labs = float(perc_of_labs)
        assert (type(perc_of_labs) == float) or (type(perc_of_labs) == type(None)), f"\n'perc_of_labs' must be a float, integer, or None type. \ncurently value is '{perc_of_labs}' and type is {type(perc_of_labs)}"
        assert (perc_of_labs > 0) & (perc_of_labs <= 1), f"\n'perc_of_labs' must be greater than 0, and less than or equal to 1. \ncurently value is {perc_of_labs}"
        print("value of perc_of_labs is acceptable")

    def validate_perc_of_data(perc_of_data):
        if type(perc_of_data) == int:
            perc_of_data = float(perc_of_data)
        assert (type(perc_of_data) == float) or (type(perc_of_data) == type(None)), f"\n'perc_of_labs' must be a float, integer, or None type. \ncurently value is '{perc_of_data}' and type is {type(perc_of_data)}"
        assert (perc_of_data > 0) & (perc_of_data <= 1), f"\n'perc_of_labs' must be greater than 0, and less than or equal to 1. \ncurently value is {perc_of_data}"
        print("value of perc_of_labs is acceptable")

    def validate_data_splits(data_splits):
        sum_of_splits = sum([val for key, val in data_splits.items()])
        assert sum_of_splits == 1.0, f"the sum of the values of data_splits must equal 1.0 \n currenlty they are {[val for key, val in data_splits.items()]} and sum up to {sum_of_splits}"
        print("values of data_splits are acceptable")

    def validate_label_file(label_filepath):
        assert type(label_filepath) == str, f"\n'label_file' must be a string type. \ncurrently it is type {type(label_filepath)}"
        print("value of label_filepath is acceptable")


    # In[ ]:


    validate_label_file_label_cols(label_file_label_cols)
    validate_perc_of_labs(perc_of_labs)
    validate_perc_of_data(perc_of_data)
    validate_data_splits(data_splits)


    # In[ ]:


    fasta_sep = '>'

    example_col_name = 'example_id'
    search_strings = ['/1', '/2']
    replace_strings = ['', '']
    split_col_name = 'split'

    non_mer_regex = '[A-Z]+[ ]{2,}.*'


    # In[ ]:


    data_splits = {x:data_splits[x] for x in sorted(data_splits, key=data_splits.get, reverse=True)}


    # In[ ]:


    unsplit_data_filepath = Path(output_folder_path) / dataset_name / f"{dataset_name}_unsplit.parquet"
    data_label_filepath = Path(output_folder_path) / dataset_name / f"{dataset_name}_orig_labels.csv"
    data_new_label_filepath = Path(output_folder_path) / dataset_name / f"{dataset_name}_labels.csv"
    vocab_filepath = Path(output_folder_path) / dataset_name / f"{dataset_name}_vocab.txt"
    data_splits_filepaths = {}
    for key, val in data_splits.items():
        data_splits_filepaths[key] = Path(output_folder_path) / dataset_name / f"{dataset_name}_{key}.parquet"
    data_splits_filepaths_csv = {}
    for key, val in data_splits.items():
        data_splits_filepaths_csv[key] = Path(output_folder_path) / dataset_name / f"{dataset_name}_{key}.csv"


    # make new raw file
    # + shuffled
    # + percent of labels
    # + precent of data
    # 

    # outline
    # + part A
    #     1. load fasta file
    #     2. format correctly
    #     3. randomly select percent
    #     4. shuffle
    #     5. save file
    # 
    # + part B
    #     6. format id correctly 
    #     7. create id file to reference 
    #     8. subset percent of labels
    #     9. save data splits
    # 
    # + part C
    #     9. preprocess dataframes
    #     10. create vocab file

    # In[ ]:


    os.system('rm -rf ./dask-worker-space')


    # In[ ]:


    cluster = LocalCUDACluster(silence_logs=50)
    client = Client(cluster)


    # In[ ]:


    print(f"the dask dashboard can be found here:\n{client.dashboard_link}\n")


    # In[ ]:


    def cull_empty_partitions(df):
        ll = list(df.map_partitions(len).compute())
        df_delayed = df.to_delayed()
        df_delayed_new = list()
        pempty = None
        for ix, n in enumerate(ll):
            if 0 == n:
                pempty = df.get_partition(ix)
            else:
                df_delayed_new.append(df_delayed[ix])
        if pempty is not None:
            df = dask_cudf.from_delayed(df_delayed_new, meta=pempty)
        return df


    # In[ ]:


    df = dask_cudf.read_csv(input_filepath,  # location of raw file
                            sep=fasta_sep,  # this is the '>' sign
                            names=raw_cols,  # column names
                            dtype=str,  # data type
                            )

    df[raw_lab_col] = df[raw_lab_col].shift()

    df = df.dropna().reset_index(drop=True)


    # In[ ]:


    if (perc_of_data != None) & (perc_of_data < 1.0):
        def subset_data(df):
            df['random_num'] = ''
            num_random = df.shape[0]
            if do_random_seed:
                np.random.seed(random_seed)
            df['random_num'] = np.random.uniform(size=num_random)
            keep_mask = df['random_num'] < perc_of_data
            df = df[keep_mask]
            df = df.drop(columns=['random_num'])
            return df
        df = df.map_partitions(subset_data)


    # In[ ]:


    if shuffle_data:
        def random_shuffle(df):
            df['random_num'] = ''
            num_random = df.shape[0]
            if do_random_seed:
                np.random.seed(random_seed)
            df['random_num'] = np.random.uniform(size=num_random)
            df = df.set_index(df['random_num'] ).sort_index().reset_index(drop=True)
            return df
        df = df.map_partitions(random_shuffle)
        df = df.drop(columns='random_num')


    # In[ ]:


    df = cull_empty_partitions(df)


    # In[ ]:


    df.to_parquet(unsplit_data_filepath)


    # In[ ]:


    del df


    # ## part 2

    # In[ ]:


    keep_cols = []
    keep_cols.append(label_file_id_col)
    for ith in label_file_label_cols:
        keep_cols.append(ith['name_col'])
        keep_cols.append(ith['code_col'])

    label_df = dask_cudf.read_csv(label_filepath, sep='\t', dtype=str, usecols=keep_cols)
    label_df.to_csv(data_label_filepath, index=False, single_file=True)
    label_df = label_df.compute().drop_duplicates()


    # In[ ]:


    # label_df


    # In[ ]:


    df = dask_cudf.read_parquet(unsplit_data_filepath)


    # In[ ]:


    # df.head()


    # In[ ]:


    df = df.assign(example_id='')
    df['example_id'] = df[label_file_id_col]

    search_strings = ['/1', '/2']
    replace_strings = ['', '']

    def extract_example(df):
        df['example_id'] = df['example_id'].str.replace(pat=search_strings, repl=replace_strings, n=None)
        return df

    df = df.map_partitions(extract_example, meta=extract_example(df.head()))


    # In[ ]:


    # df.head()


    # In[ ]:


    def extract_labels(df):
        df[raw_lab_col] = df[raw_lab_col].str.split('|').list.get(-1).str.split('-').list.get(0).str.split('_').list.get(0)
        return df

    df = df.map_partitions(extract_labels)


    # In[ ]:


    # df.head()


    # In[ ]:


    code_cols = [x['code_col'] for x in label_file_label_cols]
    label_df_merge_cols = [label_file_id_col] + code_cols
    label_df_merge_cols

    def merge_label_codes(df):
        merge_label_df = label_df[label_df_merge_cols]
        df = df.merge(merge_label_df, how='left', on=label_file_id_col)
        return df

    df = df.map_partitions(merge_label_codes, meta=merge_label_codes(df.head()))


    # In[ ]:


    # df.head()


    # In[ ]:


    # # check to see if there are any missing IDs
    def check_na(df):

        isna_mask = df.isna().all(axis=1)
        df = df[isna_mask]
        return df

    na_df = df.map_partitions(check_na, meta=check_na(df.head())).compute()
    assert na_df.shape[0] == 0, f"you have missing ID values!"


    # In[ ]:


    if (perc_of_labs != None) & (perc_of_labs < 1.0):
        label_df['random_num'] = ''
        num_random = label_df.shape[0]
        if do_random_seed:
                np.random.seed(random_seed)
        label_df['random_num'] = np.random.uniform(size=num_random)
        keep_mask = label_df['random_num'] < perc_of_labs
        sub_label_df = label_df[keep_mask].copy()
        sub_label_df = sub_label_df.drop(columns=['random_num'])
        label_df = label_df.drop(columns=['random_num'])

        def subset_labels(df):
                df = df.set_index(label_file_id_col)
                df = df.loc[sub_label_df[label_file_id_col].unique()]
                df = df.reset_index()
                return df

        df = df.map_partitions(subset_labels, meta=subset_labels(df.head(100)))

        sub_label_df.to_csv(str(data_label_filepath).replace('.csv', '_subset.csv'), index=False)




    # In[ ]:


    # df.head()


    # In[ ]:


    if shuffle_data:
        def random_shuffle(df):
            df['random_num'] = ''
            num_random = df.shape[0]
            if do_random_seed:
                np.random.seed(random_seed)
            df['random_num'] = np.random.uniform(size=num_random)
            df = df.set_index(df['random_num'] ).sort_index().reset_index(drop=True)
            return df
        df = df.map_partitions(random_shuffle)
        df = df.drop(columns='random_num')


    # In[ ]:


    # # check to see if there are any missing IDs
    def check_na(df):

        isna_mask = df.isna().all(axis=1)
        df = df[isna_mask]
        return df

    na_df = df.map_partitions(check_na, meta=check_na(df.head())).compute()
    assert na_df.shape[0] == 0, f"you have missing ID values!"


    # In[ ]:


    temp_df = df.copy().assign(count=1)
    lab_md = []
    for code_col in code_cols:
        temp2 = temp_df[[code_col] + ['count']]
        temp2 = temp2.groupby(code_col).sum()
        temp2['lab_type'] = code_col
        temp2['lab_code'] = temp2.index
        temp2 = temp2.reset_index(drop=True)
        temp2 = temp2[['lab_type', 'lab_code', 'count']]
        lab_md.append(temp2.compute())
    lab_md = cudf.concat(lab_md, ignore_index=True)


    # In[ ]:


    lab_md['lab_type'].value_counts()


    # In[ ]:


    df.head()


    # In[ ]:


    #########################################################################################################################################################
    #################### CLASS RELABELING: NEED TO UPDATE. CURRENLT HARDCODED SPECIES AND GENUS. MUST REFACTOR TO GENERALIZE #################################
    #########################################################################################################################################################
    cur_lab_df = df[code_cols].drop_duplicates().copy().compute()
    spec_rename_df = cur_lab_df[['species_code']].drop_duplicates().astype(int, errors='ignore').sort_values('species_code').reset_index(drop=True).reset_index().rename(columns={'index':'new_species_code'}).copy()
    gen_rename_df = cur_lab_df[['genus_code']].drop_duplicates().astype(int, errors='ignore').sort_values('genus_code').reset_index(drop=True).reset_index().rename(columns={'index':'new_genus_code'}).copy()
    def rename_labs(df):
        df = df.merge(spec_rename_df, how='left', on='species_code')
        df = df.merge(gen_rename_df, how='left', on='genus_code')
        df = df.drop(columns=['species_code', 'genus_code'])
        df = df.rename(columns={'new_genus_code':'genus_code', 'new_species_code':'species_code'})
        return df
    df = df.map_partitions(rename_labs, meta=rename_labs(df.head(100)))

    orig_label_df = cudf.read_csv(data_label_filepath)
    orig_label_df = df[['id']].copy().drop_duplicates().merge(orig_label_df, how='left', on='id')
    new_label_df = rename_labs(orig_label_df).dropna().compute()
    _ = new_label_df.to_csv(data_new_label_filepath, index=False)
    #########################################################################################################################################################
    #########################################################################################################################################################


    # In[ ]:


    unq_examples = df['example_id'].unique().to_frame()
    unq_examples = unq_examples.assign(split='')


    # In[ ]:


    # df.head()


    # In[ ]:


    def add_spit_val(df):
        df['random_num'] = ''
        num_random = df.shape[0]
        if do_random_seed:
            np.random.seed(random_seed)
        df['random_num'] = np.random.uniform(size=num_random)

        cur_lthresh = 0
        for i, (split_name, split_perc) in enumerate(data_splits.items()):
            if i == 0:
                # set stuff
                split_mask = df['random_num'] < split_perc
                df['split'][split_mask] = split_name
                cur_lthresh += split_perc
            else:
                cur_uthresh = cur_lthresh + split_perc
                split_mask = (df['random_num'] < cur_uthresh) & (df['random_num'] >= cur_lthresh)
                df['split'][split_mask] = split_name
                cur_lthresh += split_perc

        df = df.drop(columns=['random_num'])
        return df


    # In[ ]:


    unq_examples = unq_examples.map_partitions(add_spit_val, meta=add_spit_val(unq_examples.head()))


    # In[ ]:


    df = df.merge(unq_examples, how='left', on='example_id')


    # In[ ]:


    if shuffle_data:
        def random_shuffle(df):
            df['random_num'] = ''
            num_random = df.shape[0]
            if do_random_seed:
                np.random.seed(random_seed)
            df['random_num'] = np.random.uniform(size=num_random)
            df = df.set_index(df['random_num'] ).sort_index().reset_index(drop=True)
            return df
        df = df.map_partitions(random_shuffle)
        df = df.drop(columns='random_num')


    # In[ ]:


    # df.head(20)


    # In[ ]:


    for split_name, split_filepath in data_splits_filepaths.items():
        # print(split_name)
        def extract_split(df):
            df = df[df[split_col_name] == split_name]
            df = df.drop(columns=['example_id', 'id', 'split'])
            df = df.reset_index(drop=True)
            return df
        temp_df = df.map_partitions(extract_split, meta=extract_split(df.head()))
        temp_df = cull_empty_partitions(temp_df)
        _ = temp_df.to_parquet(split_filepath)

        # create label metadata: count of each label
        temp_df = temp_df.assign(count=1)
        lab_md_filepath = str(split_filepath).replace('.parquet', '_lab_md.csv')
        lab_md = []
        for code_col in code_cols:
            temp2 = temp_df[[code_col] + ['count']]
            temp2 = temp2.groupby(code_col).sum()
            temp2['lab_type'] = code_col
            temp2['lab_code'] = temp2.index
            temp2 = temp2.reset_index(drop=True)
            temp2 = temp2[['lab_type', 'lab_code', 'count']]
            lab_md.append(temp2.compute())
        lab_md = cudf.concat(lab_md, ignore_index=True)

        _ = lab_md.to_csv(lab_md_filepath, index=False)


    # In[ ]:


    del temp_df, unq_examples, label_df, df


    # In[ ]:


    if (perc_of_labs != None) & (perc_of_labs < 1.0):
        del sub_label_df


    # ## part 3

    # In[ ]:


    vocab_data = '\n' + '\n'.join(possible_gene_values) #+ '\n' + sos_token + '\n' + eos_token

    raw_vocab_list = vocab_data.split('\n')
    token_vocab_list = [str(x) for x in list(range(len(raw_vocab_list)))]
    # token_vocab_list = list(range(len(raw_vocab_list)))


    vocab_data = vocab_data + '\n[UNK]' + '\n[MASK]'
    open(vocab_filepath, 'w').write(vocab_data)


    # In[ ]:


    raw_vocab_list


    # In[ ]:


    token_vocab_list


    # In[ ]:


    def to_digits(df):
        df[raw_seq_col] = df[raw_seq_col].str.replace(raw_vocab_list[1::], token_vocab_list[1::], regex=False)
        return df


    # In[ ]:


    def drop_non_digits(df):
        df = df[df[raw_seq_col].str.isdigit() == True]
        return df


    # In[ ]:


    replace_gene_values = []
    func_possible_gene_values = token_vocab_list[1::].copy()
    func_possible_gene_values += ['0']
    for gene_val in func_possible_gene_values:
        replace_gene_values.append(gene_val + ' ')


    # In[ ]:


    replace_gene_values


    # In[ ]:


    func_possible_gene_values


    # In[ ]:


    def add_whitespace(df):
        df[raw_seq_col] = df[raw_seq_col].str.replace(func_possible_gene_values, replace_gene_values, regex=False)
        return df


    # In[ ]:


    def add_padding(df):
        df[raw_seq_col] = df[raw_seq_col].str.pad(width=max_input_len, side='right', fillchar='0')
        return df


    # In[ ]:


    out_label_col_names = [x['code_col'] for x in label_file_label_cols]


    # In[ ]:


    def label_to_int_dtype(df):
        for out_col in out_label_col_names:
            df[out_col] = df[out_col].astype(int)
        return df


    # In[ ]:


    def split_dna_seq(df):

        # df = add_sos_eos(df)
        df = to_digits(df)
        df = drop_non_digits(df)
        df = add_padding(df)
        df = add_whitespace(df)

        seq_ser = df[raw_seq_col].copy()

        seq_ser = seq_ser.str.split(expand=True)

        seq_ser = seq_ser.replace(' ', '')
        seq_ser = seq_ser.fillna(0)

        seq_ser = seq_ser.astype('float32')

        old_col_names = seq_ser.columns
        new_col_names = [f"{raw_seq_col}_{str(old_col_name)}" for old_col_name in old_col_names]
        seq_ser = seq_ser.rename(columns={x:y for x,y in zip(old_col_names, new_col_names)})

        for out_col in out_label_col_names:
            seq_ser[out_col] = df[out_col].copy()#.astype('int64')

        df = seq_ser

        return df


    # In[ ]:


    for split_name, split_filepath in data_splits_filepaths.items():
        out_pathname = data_splits_filepaths_csv[split_name]
        df = dask_cudf.read_parquet(split_filepath)
        df = df.map_partitions(split_dna_seq, meta=split_dna_seq(df.head(100)))
        _ = df.to_csv(out_pathname, index=False, single_file=True)
        del df
        # pass


    # In[ ]:


    print("processes finished, shutting down cluster...")


    # In[ ]:


    client.shutdown()


    # In[ ]:


    print("cluster shutdown, deleting client...")


    # In[ ]:


    del client


    # In[ ]:




if __name__ == '__main__':
    main()